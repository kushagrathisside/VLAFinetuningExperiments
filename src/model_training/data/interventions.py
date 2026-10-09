"""Turn a :class:`~model_training.config.schema.DatasetIntervention` into a real,
lineage-tracked dataset version.

Every intervention becomes an actual ``rde`` ``transform`` (episode selection)
followed by ``build_training_dataset`` run -- never a local, untraceable copy.
That means:

* ``full``       no transform step at all; builds straight off the input.
* ``fraction`` / ``reduced``    a seeded random episode draw, applied via
  ``rde.transformations.EpisodeSubsetFilter`` -- a real ``clean/vN`` with real
  dropped-episode lineage.
* ``imbalanced``  the same mechanism, selecting the *complement* of a chosen
  ``variable``/``variable_value`` (today: ``task_id``, the one grouping key
  this dataset's schema actually carries -- see
  ``model_training.data.profile``'s variation-coverage note).
* ``rebalanced``    no episode exclusion at all -- every input episode is
  kept, and the under-represented episodes are given a higher *training
  sampling weight* via ``DatasetBuilder.build``'s existing ``episode_weights``
  (the same mechanism ``rde.feedback.flywheel`` already uses for the deployment
  feedback loop). Real data, oversampled in place; nothing duplicated or
  fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rde.catalog import Catalog
from rde.config import Settings, get_settings
from rde.dataset_builder import DatasetBuilder
from rde.schemas.lineage import VersionStage
from rde.transformations import EpisodeSubsetFilter, TransformPipeline
from rde.versioning import next_version

from ..config.schema import DatasetIntervention, SplitConfig


def compute_length_bucket(episode_rows: list[dict[str, Any]]) -> dict[int, str]:
    """short/medium/long by within-dataset episode-length terciles.

    The one grouping key that is *always* available regardless of how many
    distinct tasks a dataset carries -- ``so100_pickplace`` itself has a
    single task instruction, so ``task_id``-based imbalance has nothing to
    select between there, while trajectory length always varies. Same
    bucketing method as ``model_training.data.profile``'s coverage section, so
    the two stay comparable.
    """
    if len(episode_rows) < 3:
        return {r["episode_id"]: "medium" for r in episode_rows}
    lengths = [r["num_frames"] for r in episode_rows]
    t1, t2 = np.percentile(lengths, [33.3, 66.7])
    return {
        r["episode_id"]: ("short" if r["num_frames"] <= t1 else "long" if r["num_frames"] > t2 else "medium")
        for r in episode_rows
    }


def resolve_episode_subset(
    intervention: DatasetIntervention,
    all_episode_ids: list[int],
    task_of_episode: dict[int, int | None],
    length_bucket_of_episode: dict[int, str] | None = None,
) -> tuple[list[int], str]:
    """Episode ids to keep, and the reason, for every kind but ``rebalanced``.

    An explicit ``episode_ids`` always wins when given. Otherwise the rule is
    resolved from the dataset's real per-episode metadata -- never a guess.
    """
    kind = intervention.kind
    if kind == "full":
        return sorted(all_episode_ids), "full dataset, no episode selection"
    if intervention.episode_ids is not None:
        return sorted(intervention.episode_ids), intervention.reason or f"explicit episode_ids ({kind})"

    if kind in ("fraction", "reduced"):
        frac = intervention.fraction if intervention.fraction is not None else 0.5
        rng = np.random.default_rng(intervention.seed)
        ids = sorted(all_episode_ids)
        # frac=0.0 is a legitimate, explicit request (a zero-data anchor
        # point on a data-efficiency sweep) and must be honored as 0
        # episodes, not silently floored to 1 -- only a *positive* fraction
        # that happens to round down to 0 (e.g. 1% of a 3-episode dataset)
        # gets the floor, so a real request for "some data" never silently
        # becomes "no data".
        n = round(frac * len(ids))
        if frac > 0:
            n = max(1, n)
        chosen = sorted(int(i) for i in rng.choice(ids, size=n, replace=False))
        return chosen, intervention.reason or f"seeded random {frac:.0%} draw over all episodes (seed={intervention.seed})"

    if kind == "imbalanced":
        var, val = intervention.variable, intervention.variable_value
        if var == "task_id":
            keep = sorted(e for e in all_episode_ids if task_of_episode.get(e) != val)
            return keep, intervention.reason or f"drop every episode with task_id={val!r}"
        if var == "episode_length_bucket" and length_bucket_of_episode is not None:
            keep = sorted(e for e in all_episode_ids if length_bucket_of_episode.get(e) != val)
            return keep, intervention.reason or f"drop every episode in length bucket {val!r}"
        raise ValueError(
            f"resolve_episode_subset has no automatic rule for variable={var!r}; "
            "supported: 'task_id', 'episode_length_bucket', or pass an explicit "
            "intervention.episode_ids instead"
        )

    raise ValueError(f"resolve_episode_subset cannot resolve kind={kind!r} without explicit episode_ids")


@dataclass
class InterventionResult:
    dataset_id: str
    input_version: str
    clean_version: str
    training_version: str
    kind: str
    episode_ids: list[int]
    excluded_episode_ids: list[int]
    episode_weights: dict[int, float]
    selection_reason: str
    build_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__}


def build_intervention_dataset(
    dataset_id: str,
    input_version: str,
    intervention: DatasetIntervention,
    splits: SplitConfig,
    settings: Settings | None = None,
    catalog: Catalog | None = None,
) -> InterventionResult:
    """Materialise one intervention as a real ``clean/vN`` (if any exclusion is
    needed) plus a real ``training/vN``, both queryable via ``rde version
    lineage`` afterwards."""
    settings = settings or get_settings()
    catalog = catalog or Catalog(settings.paths.catalog_db)

    manifest = catalog.get_manifest(dataset_id, input_version)
    if manifest is None:
        raise LookupError(f"unknown version: {dataset_id}@{input_version}")
    episode_rows = catalog.get_episodes(dataset_id, input_version)
    all_ids = [e["episode_id"] for e in episode_rows]
    task_of = {e["episode_id"]: e["task_id"] for e in episode_rows}
    length_bucket_of = compute_length_bucket(episode_rows)
    if not all_ids:
        raise ValueError(f"{dataset_id}@{input_version} has no episodes to build from")

    def _target_episodes(var: str | None, val: Any) -> list[int]:
        if var == "task_id":
            return [e for e in all_ids if task_of.get(e) == val]
        if var == "episode_length_bucket":
            return [e for e in all_ids if length_bucket_of.get(e) == val]
        return []

    if intervention.kind == "rebalanced":
        weights = dict(intervention.episode_weights or {})
        if not weights and intervention.variable is not None and intervention.variable_value is not None:
            target = _target_episodes(intervention.variable, intervention.variable_value)
            if not target:
                raise ValueError(
                    f"no episodes with {intervention.variable}={intervention.variable_value!r} to upweight"
                )
            weights = {e: intervention.oversample_factor for e in target}
        if not weights:
            raise ValueError(
                "rebalanced intervention resolved to empty episode_weights -- "
                "nothing would change relative to a full-data run"
            )
        subset_ids = sorted(all_ids)
        excluded: list[int] = []
        clean_version = input_version
        reason = intervention.reason or (
            f"rebalanced: oversample {len(weights)} episode(s) by weight, no exclusion"
        )
    else:
        subset_ids, reason = resolve_episode_subset(intervention, all_ids, task_of, length_bucket_of)
        if not subset_ids:
            raise ValueError(f"intervention kind={intervention.kind!r} resolved to zero episodes")
        excluded = sorted(set(all_ids) - set(subset_ids))
        weights = intervention.episode_weights or {}
        if subset_ids == sorted(all_ids):
            clean_version = input_version
        else:
            clean_version = next_version(catalog, dataset_id, VersionStage.CLEAN)
            TransformPipeline(
                [EpisodeSubsetFilter(subset_ids, reason=reason)],
                settings=settings, catalog=catalog,
            ).run(dataset_id, input_version, clean_version)

    training_version = next_version(catalog, dataset_id, VersionStage.TRAINING)
    build = DatasetBuilder(settings=settings, catalog=catalog).build(
        dataset_id, clean_version, training_version,
        splits=splits.as_fractions(), seed=splits.seed, split_mode=splits.mode,
        episode_weights=weights or None,
    )

    return InterventionResult(
        dataset_id=dataset_id,
        input_version=input_version,
        clean_version=clean_version,
        training_version=training_version,
        kind=intervention.kind,
        episode_ids=subset_ids,
        excluded_episode_ids=excluded,
        episode_weights=weights,
        selection_reason=reason,
        build_summary=build.summary(),
    )
