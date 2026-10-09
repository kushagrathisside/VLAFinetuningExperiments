"""Dataset profiling: a machine-readable snapshot of one dataset version.

Reuses ``rde.exploration.queries.DatasetExplorer`` for everything it already
computes (overview, duration distribution, task summary, modality summary,
per-dimension state/action statistics, QC summary, lineage) rather than
re-deriving any of it -- this module's only real job is packaging those into
the profile shape and adding the two things
``DatasetExplorer`` does not compute: coverage/diversity scores and the
train/val/test split distribution (which only exists on a *training* version).

Coverage methodology, stated once here rather than per-field: a dimension's
"coverage" is the **normalized Shannon entropy** of its observed value
distribution -- 0 when every sample lands in one bin (no coverage), 1 when
observations are spread evenly across all bins used. This is a real,
computable proxy for "how much of the observed range gets used", not a claim
about coverage of the true task space, which cannot be measured from data
alone. Where a requested dimension has no honest computation available
from this dataset's actual schema (visual diversity needs an embedding model;
this profiler does not run one), it is reported as ``"unavailable"`` with the
reason, never approximated silently.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Any

from rde.catalog import Catalog
from rde.config import Settings, get_settings
from rde.engine import DataEngine
from rde.exploration.queries import DatasetExplorer

_HIST_BINS = 10


def _shannon_entropy_bits(counts: list[int]) -> float | None:
    total = sum(counts)
    if total <= 0:
        return None
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def _normalized_entropy(counts: list[int]) -> float | None:
    """0 (all mass in one bucket) .. 1 (uniform across every non-empty bucket)."""
    h = _shannon_entropy_bits(counts)
    if h is None:
        return None
    k = sum(1 for c in counts if c > 0)
    if k <= 1:
        return 0.0
    return round(h / math.log2(k), 4)


@dataclass
class DatasetProfile:
    dataset_id: str
    version: str
    parent_version: str | None
    generated_at: str
    episode_count: int
    frame_count: int
    task_count: int
    tasks: list[dict[str, Any]]
    episode_length: dict[str, Any]
    action: dict[str, Any]
    state: dict[str, dict[str, Any]]
    modalities: list[dict[str, Any]]
    image: dict[str, Any]
    language: dict[str, Any]
    qc: list[dict[str, Any]]
    splits: dict[str, int]
    missing_data: dict[str, Any]
    duplicate_summary: dict[str, Any]
    coverage: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "parent_version": self.parent_version,
            "generated_at": self.generated_at,
            "episode_count": self.episode_count,
            "frame_count": self.frame_count,
            "task_count": self.task_count,
            "tasks": self.tasks,
            "episode_length": self.episode_length,
            "action": self.action,
            "state": self.state,
            "modalities": self.modalities,
            "image": self.image,
            "language": self.language,
            "qc": self.qc,
            "splits": self.splits,
            "missing_data": self.missing_data,
            "duplicate_summary": self.duplicate_summary,
            "coverage": self.coverage,
            "notes": self.notes,
        }


def _dim_coverage(explorer: DatasetExplorer, dataset_id: str, version: str,
                   column: str, index: int, lo: float, hi: float) -> dict[str, Any]:
    """Normalized-entropy coverage for one scalar dimension, via one DuckDB pass."""
    if hi <= lo:
        return {"normalized_entropy": None, "reason": "zero-range (constant) dimension"}
    col = '"' + column.replace('"', '""') + '"'
    sql = (
        f"SELECT LEAST({_HIST_BINS - 1}, GREATEST(0, "
        f"CAST(FLOOR({_HIST_BINS} * ({col}[{index + 1}] - {lo}) / ({hi} - {lo})) AS INTEGER))) "
        f"AS bucket, COUNT(*) AS n FROM frames GROUP BY 1"
    )
    try:
        rows = explorer.sql(dataset_id, version, sql)
    except Exception as exc:  # pragma: no cover - defensive, DuckDB errors are dataset-shape specific
        return {"normalized_entropy": None, "reason": f"histogram query failed: {exc}"}
    counts = [0] * _HIST_BINS
    for r in rows:
        b = r["bucket"]
        if b is not None:
            counts[int(b)] += int(r["n"])
    return {
        "bins": _HIST_BINS, "bin_counts": counts,
        "normalized_entropy": _normalized_entropy(counts),
    }


def build_dataset_profile(
    dataset_id: str,
    version: str,
    settings: Settings | None = None,
    catalog: Catalog | None = None,
) -> DatasetProfile:
    """Profile one dataset version. Read-only; writes nothing to the catalog."""
    settings = settings or get_settings()
    catalog = catalog or Catalog(settings.paths.catalog_db)
    explorer = DatasetExplorer(settings=settings, catalog=catalog)

    overview = explorer.overview(dataset_id, version)
    durations = explorer.duration_distribution(dataset_id, version)
    tasks = explorer.task_summary(dataset_id, version)
    modalities = explorer.modality_summary(dataset_id, version)
    dims = explorer.dimensionality(dataset_id, version)
    qc = explorer.qc_summary(dataset_id, version)
    lineage = explorer.lineage(dataset_id, version)

    parent_version = lineage["chain"][-1]["inputs"][0] if lineage["chain"] and lineage["chain"][-1]["inputs"] else None

    manifest = catalog.get_manifest(dataset_id, version)
    contract = manifest.attributes.get("training_contract") if manifest else None
    if not contract:
        from rde.dataset_builder.builder import build_training_contract

        contract = build_training_contract(manifest) if manifest else {"observation": {}, "action": {}}

    act_key = contract.get("action", {}).get("key")
    action_stats = dims.get(act_key, {}) if act_key else {}
    state_keys = [k for k, v in contract.get("observation", {}).items() if v.get("plane") == "inline"]
    state_stats = {k: dims.get(k, {}) for k in state_keys}
    image_keys = [k for k, v in contract.get("observation", {}).items() if v.get("plane") == "external"]
    image_shapes = {k: contract["observation"][k].get("shape") for k in image_keys}

    notes: list[str] = []

    # -- splits: only a training/vN version carries split membership ---------
    splits: dict[str, int] = {}
    if manifest is not None:
        engine = DataEngine(settings)
        try:
            with engine.get_dataset(dataset_id, version) as view:
                raw_splits = view.splits()
            # DatasetView.splits() reports an "unassigned" bucket for episodes
            # with no split attribute at all -- a clean/vN version (pre-build)
            # is entirely "unassigned", which means no real split exists yet,
            # not that there is one named "unassigned".
            if set(raw_splits) - {"unassigned"}:
                splits = raw_splits
        except Exception as exc:
            notes.append(f"could not resolve splits for {dataset_id}@{version}: {exc}")
    if not splits:
        notes.append(
            f"{version!r} carries no split assignment (it is not a training/vN version) -- "
            "splits reported empty, not fabricated"
        )

    # -- language / instruction diversity ------------------------------------
    task_episode_counts = [t["episodes"] for t in tasks]
    language = {
        "distinct_instructions": len(tasks),
        "episodes_per_task": {str(t["task_id"]): t["episodes"] for t in tasks},
        "diversity_normalized_entropy": _normalized_entropy(task_episode_counts),
    }

    # -- coverage --------------------------------------------------------------
    action_coverage: dict[str, Any] = {"per_dimension": [], "mean_normalized_entropy": None}
    if act_key and action_stats.get("per_dimension"):
        per_dim = []
        for i, d in enumerate(action_stats["per_dimension"]):
            per_dim.append({"index": i, "name": d.get("name"),
                             **_dim_coverage(explorer, dataset_id, version, act_key, i, d["min"], d["max"])})
        action_coverage["per_dimension"] = per_dim
        ents = [d["normalized_entropy"] for d in per_dim if d["normalized_entropy"] is not None]
        action_coverage["mean_normalized_entropy"] = round(sum(ents) / len(ents), 4) if ents else None
    else:
        action_coverage["reason"] = "no action modality resolved from the training contract"

    state_coverage: dict[str, Any] = {}
    for k in state_keys:
        st = state_stats.get(k, {})
        per_dim = []
        for i, d in enumerate(st.get("per_dimension", [])):
            per_dim.append({"index": i, "name": d.get("name"),
                             **_dim_coverage(explorer, dataset_id, version, k, i, d["min"], d["max"])})
        ents = [d["normalized_entropy"] for d in per_dim if d["normalized_entropy"] is not None]
        state_coverage[k] = {
            "per_dimension": per_dim,
            "mean_normalized_entropy": round(sum(ents) / len(ents), 4) if ents else None,
        }

    variation_note = (
        "dataset schema (rde.schemas.core.Episode) carries no explicit variation/"
        "condition field -- task_id and episode-length bucket are the only "
        "real per-episode grouping keys available, so they are reported as the "
        "candidate variation proxies rather than an invented 'variation' axis"
    )
    length_frames = durations.get("frames", {}) if durations else {}
    coverage = {
        "methodology": (
            "normalized Shannon entropy of the observed value distribution per "
            f"dimension, {_HIST_BINS} equal-width bins; 0 = all mass in one bin, "
            "1 = uniform across every non-empty bin. Measures how evenly the "
            "*observed* range is used, not coverage of any true task space."
        ),
        "task_coverage": {
            "distinct_tasks": len(tasks),
            "normalized_entropy": _normalized_entropy(task_episode_counts),
        },
        "variation_coverage": {
            "note": variation_note,
            "task_distribution": {str(t["task_id"]): t["episodes"] for t in tasks},
            "episode_length_histogram": durations.get("histogram", []),
        },
        "state_space_coverage": state_coverage,
        "action_space_coverage": action_coverage,
        "trajectory_length_coverage": durations,
        "instruction_diversity": language,
        "visual_diversity": {
            "status": "unavailable",
            "reason": "requires an image embedding/similarity model; this profiler computes "
                      "only what the canonical numeric schema and QC records support",
        },
        "episode_diversity": {
            "status": "partial",
            "distinct_task_count": len(tasks),
            "episode_length_cv": (
                round(length_frames.get("mean", 0) and
                      (float(durations["duration_s"]["max"] - durations["duration_s"]["min"])
                       / max(durations["duration_s"]["mean"], 1e-9)), 4)
                if durations else None
            ),
            "reason_partial": "limited to task-id and length variability; no trajectory-embedding "
                              "clustering is computed",
        },
    }

    missing_data = {
        "source": "rde QC finding records (qc_records) for this version -- only failing/warning "
                  "checks are persisted as rows (passing checks leave no row), so a check's "
                  "absence below means either it found nothing or has not been run against "
                  f"{dataset_id}@{version} specifically; cross-check with `rde obs quality`",
        "episodes_with_missing_or_discontinuous_frames": explorer.episodes_with_missing_frames(dataset_id, version),
    }
    duplicate_findings = [r for r in qc if r.get("check_name") == "temporal.duplicates"]
    duplicate_summary = {
        "source": missing_data["source"],
        "temporal_duplicates_findings": duplicate_findings,
    }

    return DatasetProfile(
        dataset_id=dataset_id,
        version=version,
        parent_version=parent_version,
        generated_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        episode_count=overview.episodes,
        frame_count=overview.frames,
        task_count=overview.tasks,
        tasks=tasks,
        episode_length=durations,
        action={"key": act_key, **action_stats},
        state=state_stats,
        modalities=modalities,
        image={"cameras": len(image_keys), "camera_keys": image_keys, "shapes": image_shapes},
        language=language,
        qc=qc,
        splits=splits,
        missing_data=missing_data,
        duplicate_summary=duplicate_summary,
        coverage=coverage,
        notes=notes,
    )
