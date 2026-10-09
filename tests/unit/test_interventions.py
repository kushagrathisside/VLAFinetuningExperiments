from __future__ import annotations

import pytest
from rde.catalog import Catalog
from rde.engine import DataEngine

from model_training.config.schema import DatasetIntervention, SplitConfig
from model_training.data import build_intervention_dataset, resolve_episode_subset, verify_split_integrity


def _splits():
    return SplitConfig(train=0.6, val=0.2, test=0.2, seed=1, mode="quantile")


def test_full_intervention_keeps_every_episode(settings, catalog, clean_version):
    res = build_intervention_dataset(
        "so100_test", clean_version, DatasetIntervention(kind="full"), _splits(),
        settings=settings, catalog=catalog,
    )
    assert res.clean_version == clean_version  # no transform step needed
    assert len(res.episode_ids) == 8
    assert res.excluded_episode_ids == []

    view = DataEngine(settings).get_dataset("so100_test", res.training_version)
    report = verify_split_integrity(view)
    view.close()
    assert report.ok
    assert sum(report.split_sizes.values()) == 8


def test_reduced_intervention_selects_half_and_records_lineage(settings, catalog, clean_version):
    intervention = DatasetIntervention(kind="reduced", fraction=0.5, seed=3)
    res = build_intervention_dataset(
        "so100_test", clean_version, intervention, _splits(), settings=settings, catalog=catalog,
    )
    assert len(res.episode_ids) == 4
    assert len(res.excluded_episode_ids) == 4
    assert res.clean_version != clean_version  # a real new version was built

    runs = catalog.get_runs("so100_test")
    transform_run = next(r for r in runs if r.output_version == res.clean_version)
    assert set(transform_run.output_episode_ids) == set(res.episode_ids)
    assert sorted(transform_run.dropped_episode_ids) == sorted(res.excluded_episode_ids)


def test_reduced_intervention_is_deterministic_given_seed(settings, catalog, clean_version):
    a = build_intervention_dataset(
        "so100_test", clean_version, DatasetIntervention(kind="reduced", fraction=0.5, seed=9),
        _splits(), settings=settings, catalog=catalog,
    )
    b = build_intervention_dataset(
        "so100_test", clean_version, DatasetIntervention(kind="reduced", fraction=0.5, seed=9),
        _splits(), settings=settings, catalog=catalog,
    )
    assert a.episode_ids == b.episode_ids


def test_imbalanced_intervention_drops_one_task(settings, catalog, clean_version):
    rows = catalog.get_episodes("so100_test", clean_version)
    task_id = rows[0]["task_id"]
    intervention = DatasetIntervention(kind="imbalanced", variable="task_id", variable_value=task_id)
    res = build_intervention_dataset(
        "so100_test", clean_version, intervention, _splits(), settings=settings, catalog=catalog,
    )
    kept_tasks = {r["task_id"] for r in catalog.get_episodes("so100_test", res.clean_version)}
    assert task_id not in kept_tasks


def test_rebalanced_intervention_keeps_all_episodes_and_sets_weights(settings, catalog, clean_version):
    rows = catalog.get_episodes("so100_test", clean_version)
    task_id = rows[0]["task_id"]
    intervention = DatasetIntervention(
        kind="rebalanced", variable="task_id", variable_value=task_id, oversample_factor=4.0,
    )
    res = build_intervention_dataset(
        "so100_test", clean_version, intervention, _splits(), settings=settings, catalog=catalog,
    )
    assert res.clean_version == clean_version  # nothing excluded
    assert len(res.episode_ids) == 8
    assert all(w == 4.0 for w in res.episode_weights.values())
    assert set(res.episode_weights) == {r["episode_id"] for r in rows if r["task_id"] == task_id}

    # the weight must actually reach the training version's episode attributes
    manifest = catalog.get_manifest("so100_test", res.training_version)
    assert manifest.statistics["weighted_episodes"]


def test_compute_length_bucket_splits_into_terciles():
    """Direct test of the bucketing math -- the fixture dataset used elsewhere
    in this file has uniform-length episodes (real datasets rarely do; the
    real so100_pickplace *is* single-task, which is exactly why this variable
    exists as an alternative to task_id)."""
    from model_training.data.interventions import compute_length_bucket

    rows = [{"episode_id": i, "num_frames": n} for i, n in enumerate([10, 12, 20, 21, 30, 31, 40, 42, 50])]
    buckets = compute_length_bucket(rows)
    assert set(buckets.values()) == {"short", "medium", "long"}
    assert buckets[0] == "short"
    assert buckets[8] == "long"


def test_rebalanced_intervention_by_length_bucket(settings, catalog, clean_version):
    # The synthetic fixture's episodes are all the same length, so every
    # episode buckets to "short" -- upweighting "short" therefore upweights
    # everything, which is still a valid (if degenerate) exercise of the
    # rebalanced code path end to end.
    intervention = DatasetIntervention(
        kind="rebalanced", variable="episode_length_bucket", variable_value="short", oversample_factor=5.0,
    )
    res = build_intervention_dataset(
        "so100_test", clean_version, intervention, _splits(), settings=settings, catalog=catalog,
    )
    assert len(res.episode_ids) == 8
    assert all(w == 5.0 for w in res.episode_weights.values())


def test_custom_intervention_requires_episode_ids():
    with pytest.raises(Exception):
        DatasetIntervention(kind="custom")


def test_resolve_episode_subset_unknown_variable_raises():
    intervention = DatasetIntervention(kind="imbalanced", variable="camera_condition", variable_value="dim")
    with pytest.raises(ValueError, match="no automatic rule"):
        resolve_episode_subset(intervention, [1, 2, 3], {1: 0, 2: 0, 3: 1})
