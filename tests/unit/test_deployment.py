from __future__ import annotations

import json

import pytest

from model_training.deployment import DeployedModel, DeploymentHealthError, ModelArtifact, ModelRegistry


def _make_checkpoint(tmp_path, name="ckpt"):
    ckpt = tmp_path / name
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    (ckpt / "policy_preprocessor.json").write_text("{}")
    (ckpt / "policy_postprocessor.json").write_text("{}")
    (ckpt / "model.safetensors").write_bytes(b"")
    (ckpt / "rde_checkpoint_meta.json").write_text(
        json.dumps({"checkpoint_source": "lerobot/smolvla_base", "dataset_id": "d", "dataset_version": "v"})
    )
    return ckpt


_GOOD_CONTRACT = {
    "action": {"key": "action.joint", "shape": [6]},
    "observation": {
        "state.joint": {"plane": "inline", "shape": [6], "normalization": {"scheme": "mean_std", "mean": [0]*6, "std": [1]*6}},
        "rgb.top": {"plane": "external", "shape": [224, 224, 3]},
    },
}


def test_health_check_passes_on_well_formed_checkpoint_and_contract(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    model = DeployedModel(str(ckpt), _GOOD_CONTRACT)
    result = model.health_check()
    assert result["status"] == "ok"
    assert result["cameras"] == ["rgb.top"]


def test_health_check_fails_loudly_on_missing_checkpoint_file(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    (ckpt / "policy_postprocessor.json").unlink()
    model = DeployedModel(str(ckpt), _GOOD_CONTRACT)
    with pytest.raises(ValueError, match="incomplete"):
        model.health_check()


def test_health_check_fails_on_missing_camera(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    contract = {**_GOOD_CONTRACT, "observation": {"state.joint": _GOOD_CONTRACT["observation"]["state.joint"]}}
    model = DeployedModel(str(ckpt), contract)
    with pytest.raises(DeploymentHealthError, match="camera"):
        model.health_check()


def test_health_check_fails_on_missing_state_normalization(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    contract = {
        **_GOOD_CONTRACT,
        "observation": {
            "state.joint": {"plane": "inline", "shape": [6]},  # no normalization
            "rgb.top": _GOOD_CONTRACT["observation"]["rgb.top"],
        },
    }
    model = DeployedModel(str(ckpt), contract)
    with pytest.raises(DeploymentHealthError, match="normalization"):
        model.health_check()


def test_health_check_fails_on_missing_action_shape(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    contract = {**_GOOD_CONTRACT, "action": {"key": "action.joint"}}
    model = DeployedModel(str(ckpt), contract)
    with pytest.raises(DeploymentHealthError, match="action shape"):
        model.health_check()


def test_model_registry_round_trips(tmp_path):
    registry = ModelRegistry(tmp_path / "models.jsonl")
    artifact = ModelArtifact(
        model_id="smolvla-exp1", model_version="final", artifact_path=str(tmp_path / "ckpt"),
        architecture="smolvla", pretrained_model="lerobot/smolvla_base", training_run_id="run1",
        experiment_id="exp1", dataset_id="so100_pickplace", dataset_version="training/v3",
        observation_schema={}, action_schema={}, embodiment="so100",
        dependencies={"torch": "2.10.0"}, git_commit={}, created_at="2026-01-01T00:00:00Z",
        metrics={"mse": 0.01},
    )
    registry.register(artifact)
    got = registry.get("smolvla-exp1")
    assert got is not None
    assert got["dataset_version"] == "training/v3"
    assert registry.for_experiment("exp1")[0]["model_id"] == "smolvla-exp1"
    assert registry.get("does_not_exist") is None
