"""Integration tests: DataEngine -> DatasetView -> adapter -> model
-> Prediction -> Evaluator -> metrics -> feedback artifact, exercised as a
chain rather than unit-by-unit. Uses the baseline models (no torch needed) so
this suite runs in the light `.venv`; the SmolVLA-specific chain is exercised
by tests/smoke/test_smolvla_smoke.py under `.venv-vla`.
"""

from __future__ import annotations

from rde.engine import DataEngine
from rde.evaluation import Evaluator
from rde.model_adapters import StateAdapter, build_model
from rde.model_adapters.base import Prediction

from model_training.data import build_dataset_profile, verify_split_integrity
from model_training.evaluation import stratified_evaluate
from model_training.feedback import build_feedback_artifact


def test_dataset_to_adapter_to_model_to_prediction(settings, catalog, training_version):
    view = DataEngine(settings).get_dataset("so100_test", training_version)
    try:
        adapter = StateAdapter(view.contract)
        model, _ = build_model("nearest_neighbor", view.contract)
        model.fit(view, split="train")

        samples = []
        for s in view.iter_samples(split="test"):
            samples.append(s)
            if len(samples) >= 4:
                break
        model_input = adapter.adapt(samples)
        prediction = model.timed_predict(model_input)

        assert isinstance(prediction, Prediction)
        assert prediction.actions.shape == (4, model_input.tensors["state"].shape[1])
        assert prediction.confidence.shape == (4,)
        assert prediction.frame_ids == [s.frame_id for s in samples]
    finally:
        view.close()


def test_prediction_to_evaluator_to_metrics_to_feedback_artifact(settings, catalog, training_version):
    view = DataEngine(settings).get_dataset("so100_test", training_version)
    try:
        model, adapter = build_model("nearest_neighbor", view.contract)
        model.fit(view, split="train")
        result = Evaluator().evaluate(view, model, adapter, split="test", persist=False)
        assert result.frames_evaluated > 0

        artifact = build_feedback_artifact(
            "exp-integration-test", "nearest_neighbor", "so100_test", training_version,
            result, settings=settings, catalog=catalog,
        )
        assert artifact["dataset_version"] == training_version
        assert "recommended_next_action" in artifact
        assert isinstance(artifact["candidate_explanations"], list)
        # every ranked review must trace back to a real episode in this view
        real_ids = {e.episode_id for e in view.episodes}
        for r in artifact["ranked_episode_reviews"]:
            assert r["episode_id"] in real_ids
    finally:
        view.close()


def test_split_integrity_then_profile_then_stratified_eval(settings, catalog, training_version):
    """The pre-training gate (split integrity) and the post-training analysis
    (stratified evaluation) both operate correctly on the same real view."""
    view = DataEngine(settings).get_dataset("so100_test", training_version)
    try:
        report = verify_split_integrity(view)
        assert report.ok

        profile = build_dataset_profile("so100_test", training_version, settings=settings, catalog=catalog)
        assert profile.splits == report.split_sizes

        model, adapter = build_model("mean_action", view.contract)
        model.fit(view, split="train")
        results = stratified_evaluate(view, model, adapter, base_split="test", slices=["per_task"])
        assert "overall" in results
    finally:
        view.close()
