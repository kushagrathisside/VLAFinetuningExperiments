from __future__ import annotations

from rde.engine import DataEngine
from rde.model_adapters import StateAdapter, build_model

from model_training.evaluation import (
    episode_metrics,
    error_distribution,
    evaluate_baselines,
    stratified_evaluate,
    temporal_error,
)


def test_evaluate_baselines_returns_both_and_knn_improves_on_mean(settings, catalog, training_version):
    view = DataEngine(settings).get_dataset("so100_test", training_version)
    results = evaluate_baselines(view, ["mean_action", "nearest_neighbor"], split="test")
    view.close()
    assert set(results) == {"mean_action", "nearest_neighbor"}
    assert results["nearest_neighbor"].metrics["mse"] <= results["mean_action"].metrics["mse"]


def test_evaluate_baselines_rejects_unknown_model(settings, catalog, training_version):
    view = DataEngine(settings).get_dataset("so100_test", training_version)
    try:
        import pytest

        with pytest.raises(KeyError):
            evaluate_baselines(view, ["not_a_real_model"])
    finally:
        view.close()


def test_error_distribution_and_episode_metrics(settings, catalog, training_version):
    view = DataEngine(settings).get_dataset("so100_test", training_version)
    model, adapter = build_model("nearest_neighbor", view.contract)
    model.fit(view, split="train")
    from rde.evaluation import Evaluator

    result = Evaluator().evaluate(view, model, adapter, split="test", persist=False)
    dist = error_distribution(result)
    view.close()
    assert dist["median"] <= dist["p90"] <= dist["p95"] <= dist["p99"]

    eps = episode_metrics(result)
    assert eps, "expected at least one episode in the test split"
    assert all(e["mse"] >= 0 for e in eps)


def test_temporal_error_buckets_progress(settings, catalog, training_version):
    view = DataEngine(settings).get_dataset("so100_test", training_version)
    model, adapter = build_model("mean_action", view.contract)
    model.fit(view, split="train")
    from rde.evaluation import Evaluator

    result = Evaluator().evaluate(view, model, adapter, split="test", persist=False)
    temporal = temporal_error(result, view)
    view.close()
    assert temporal["buckets"] == 10
    assert len(temporal["mean_error_by_progress_bucket"]) == 10
    assert sum(temporal["n_by_bucket"]) == result.frames_evaluated


def test_stratified_evaluate_per_task_and_length_bucket(settings, catalog, training_version):
    view = DataEngine(settings).get_dataset("so100_test", training_version)
    model, adapter = build_model("mean_action", view.contract)
    model.fit(view, split="train")
    out = stratified_evaluate(
        view, model, adapter, base_split="test", slices=["per_task", "per_length_bucket"]
    )
    view.close()
    assert "overall" in out
    assert any(k.startswith("task_") for k in out)
    # Every sliced result's episodes must be a subset of the overall split's episodes.
    overall_eps = {o.episode_id for o in out["overall"].outcomes}
    for name, res in out.items():
        if name == "overall":
            continue
        assert {o.episode_id for o in res.outcomes} <= overall_eps


def test_stratified_evaluate_rejects_unknown_slice(settings, catalog, training_version):
    import pytest

    view = DataEngine(settings).get_dataset("so100_test", training_version)
    model, adapter = build_model("mean_action", view.contract)
    model.fit(view, split="train")
    with pytest.raises(ValueError, match="unknown evaluation slice"):
        stratified_evaluate(view, model, adapter, slices=["variation"])
    view.close()
