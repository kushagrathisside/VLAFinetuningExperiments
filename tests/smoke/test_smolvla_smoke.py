"""Smoke test: tiny synthetic dataset, a handful of real SmolVLA
fine-tuning steps -- proves dataset resolution, contract resolution, model
loading, batch construction, forward pass, loss computation, checkpoint
save/reload, evaluation and report generation all work end to end.

Requires the `vla` extra (torch + lerobot) and, the first time, network access
to fetch the base checkpoint from the Hugging Face Hub. Skipped with a clear
reason when the dependency is unavailable -- never replaced with a stub.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")

from model_training.config.schema import ExperimentConfig
from model_training.experiments import ExperimentRunner
from rde.model_adapters.models import _dataset_stats_from_contract, _policy_features_from_contract


def _contract(*, state_modalities: dict[str, int], action_dim: int, cameras: dict[str, tuple]) -> dict:
    """A minimal training-contract fixture shaped like build_training_contract's
    real output, for exercising the cross-embodiment feature/stats builders
    without needing a full ingested dataset."""
    observation = {}
    for name, dim in state_modalities.items():
        observation[name] = {
            "key": name, "plane": "inline", "shape": [dim],
            "normalization": {"mean": [float(i) for i in range(dim)], "std": [1.0] * dim},
        }
    for name, shape in cameras.items():
        observation[name] = {"key": name, "plane": "external", "shape": list(shape)}
    return {
        "observation": observation,
        "action": {
            "key": "action.joint", "shape": [action_dim],
            "normalization": {"mean": [0.0] * action_dim, "std": [1.0] * action_dim},
        },
    }


def test_policy_features_concatenate_every_inline_modality_like_state_adapter_does():
    """aloha_coffee has two inline modalities (state.joint, force.effort), both
    14-dim; StateAdapter.adapt concatenates every inline modality into one
    vector (rde/model_adapters/base.py's state_keys is plane=="inline", not
    kind=="robot_state") -- the feature builder must declare the same total
    width, or normalization later mismatches the tensor StateAdapter actually
    produces (the exact failure this test guards against)."""
    contract = _contract(
        state_modalities={"state.joint": 14, "force.effort": 14}, action_dim=14,
        cameras={"rgb.cam_high": (480, 640, 3)},
    )
    input_features, output_features = _policy_features_from_contract(contract)
    assert input_features["observation.state"].shape == (28,)
    assert output_features["action"].shape == (14,)


def test_policy_features_single_inline_modality_matches_so100_shape():
    contract = _contract(
        state_modalities={"state.joint": 6}, action_dim=6,
        cameras={"rgb.top": (224, 224, 3), "rgb.wrist": (224, 224, 3)},
    )
    input_features, _ = _policy_features_from_contract(contract)
    assert input_features["observation.state"].shape == (6,)
    assert len(input_features) == 3  # state + 2 cameras


def test_dataset_stats_concatenate_in_the_same_order_as_the_features():
    contract = _contract(
        state_modalities={"state.joint": 2, "force.effort": 2}, action_dim=3,
        cameras={"rgb.cam_high": (480, 640, 3)},
    )
    input_features, output_features = _policy_features_from_contract(contract)
    stats = _dataset_stats_from_contract(contract, input_features, output_features)
    # state.joint contributes means [0, 1], force.effort contributes [0, 1] again
    assert stats["observation.state"]["mean"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert stats["action"]["mean"].tolist() == [0.0, 0.0, 0.0]


def test_smolvla_end_to_end_smoke(settings, catalog, clean_version, tmp_path):
    cfg = ExperimentConfig.model_validate({
        "experiment_name": "smolvla_smoke",
        "hypothesis": "smoke test only -- proves the pipeline runs, not a real result",
        "dataset": {"dataset_id": "so100_test", "version": clean_version},
        "splits": {"train": 0.6, "val": 0.2, "test": 0.2, "seed": 1},
        "training": {
            "model": "smolvla", "pretrained_checkpoint": "lerobot/smolvla_base",
            "steps": 4, "batch_size": 2, "checkpoint_every": 2, "validate_every": 2,
            "val_batches": 1, "keep_last_n_checkpoints": 1,
        },
        # A small, non-divisor batch_size forces multiple predict() calls of
        # different sizes within one evaluation (e.g. 3,3,3,...,2) -- the
        # exact condition that exposed the SmolVLAPolicy.predict() action-queue
        # bug (predict() must never reuse a cached action chunk across
        # unrelated batches; see rde/model_adapters/models.py::predict).
        "evaluation": {"baseline_models": ["mean_action"], "max_frames": 20, "batch_size": 3, "slices": []},
        "tags": {"purpose": "smoke_test"},
    })
    runner = ExperimentRunner(
        settings=settings, catalog=catalog,
        runs_dir=tmp_path / "runs", index_dir=tmp_path / "experiments",
    )
    result = runner.run(cfg)
    run_dir = result.run_dir

    # -- checkpointing --------------------------------------------------------
    assert (run_dir / "checkpoints" / "final").exists()
    outcome = json.loads((run_dir / "training_outcome.json").read_text())
    assert outcome["steps_completed"] == 4
    assert outcome["final_train_loss"] is not None
    assert outcome["final_train_loss"] == outcome["final_train_loss"]  # not NaN

    # -- checkpoint reload ------------------------------------------------------
    from rde.model_adapters.models import SmolVLAPolicy

    reloaded = SmolVLAPolicy.from_checkpoint(outcome["best_checkpoint"])
    assert reloaded.describe()["loaded"] is True

    # -- evaluation / reports ------------------------------------------------------
    assert (run_dir / "evaluation" / "test_overall.json").exists()
    assert (run_dir / "evaluation" / "baselines.json").exists()
    assert (run_dir / "model_artifact.json").exists()
    assert (run_dir / "reports" / "experiment_report.md").exists()
    assert (run_dir / "reports" / "model_card.md").exists()
    assert (run_dir / "failures" / "feedback_artifact.json").exists()
    assert (run_dir / "predictions" / "test_outcomes.jsonl").exists()

    test_overall = json.loads((run_dir / "evaluation" / "test_overall.json").read_text())
    assert "mse" in test_overall["metrics"]
