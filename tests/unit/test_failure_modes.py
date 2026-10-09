"""Explicit failure-mode tests: every one of these must raise a
specific, actionable error -- never degrade silently or return a plausible-
looking empty/default result."""

from __future__ import annotations

import pytest
from rde.engine import DataEngine

from model_training.config.schema import DatasetIntervention, SplitConfig
from model_training.data import build_dataset_profile, build_intervention_dataset, resolve_episode_subset
from model_training.deployment import DeployedModel, DeploymentHealthError


def test_invalid_dataset_raises(settings, catalog):
    with pytest.raises(LookupError):
        DataEngine(settings).get_dataset("does_not_exist", "training/v1")


def test_invalid_dataset_version_raises(settings, catalog, clean_version):
    with pytest.raises(LookupError):
        DataEngine(settings).get_dataset("so100_test", "training/v999")


def test_profile_of_unknown_version_raises(settings, catalog):
    with pytest.raises(LookupError):
        build_dataset_profile("so100_test", "raw/v999", settings=settings, catalog=catalog)


def test_intervention_on_unknown_input_version_raises(settings, catalog):
    with pytest.raises(LookupError):
        build_intervention_dataset(
            "so100_test", "clean/v999", DatasetIntervention(kind="full"), SplitConfig(),
            settings=settings, catalog=catalog,
        )


def test_unknown_dataset_variation_raises(settings, catalog, clean_version):
    intervention = DatasetIntervention(kind="imbalanced", variable="camera_condition", variable_value="dim")
    with pytest.raises(ValueError, match="no automatic rule"):
        build_intervention_dataset(
            "so100_test", clean_version, intervention, SplitConfig(), settings=settings, catalog=catalog,
        )


def test_reduced_intervention_at_zero_fraction_is_honored(settings, catalog, clean_version):
    """fraction=0.0 is an explicit, legitimate request (a zero-data anchor
    point on a data-efficiency sweep) and must be honored as 0 episodes, not
    silently floored to 1 -- flooring would corrupt exactly the sweep point
    meant to measure "no training data" performance. A genuinely positive
    fraction that happens to round down to 0 still gets floored to 1."""
    intervention = DatasetIntervention(kind="fraction", fraction=0.0)
    ids, _ = resolve_episode_subset(intervention, [1, 2, 3, 4], {})
    assert ids == []

    tiny = DatasetIntervention(kind="fraction", fraction=0.01)
    ids, _ = resolve_episode_subset(tiny, [1, 2, 3], {})
    assert len(ids) == 1


def test_missing_vla_dependency_reported_actionably():
    """rde's own contract: an unavailable optional model fails loudly."""
    from rde.model_adapters import SmolVLAPolicy

    policy = SmolVLAPolicy()
    try:
        policy.load()
    except (ImportError, RuntimeError, OSError) as exc:
        msg = str(exc).lower()
        assert "smolvla" in msg or "lerobot" in msg or "torch" in msg
    else:
        pytest.skip("torch/lerobot are installed in this environment -- load() succeeded, nothing to test")


def test_corrupted_checkpoint_directory_rejected(tmp_path):
    from rde.model_adapters.models import SmolVLAPolicy

    ckpt = tmp_path / "corrupted"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")  # only one of five required files
    with pytest.raises(ValueError, match="incomplete"):
        SmolVLAPolicy.verify_checkpoint(ckpt)


def test_invalid_deployment_contract_rejected(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
        (ckpt / name).write_text("{}")
    (ckpt / "model.safetensors").write_bytes(b"")
    (ckpt / "rde_checkpoint_meta.json").write_text("{}")

    model = DeployedModel(str(ckpt), contract={"action": {}, "observation": {}})
    with pytest.raises(DeploymentHealthError):
        model.health_check()


def test_split_config_rejects_fractions_not_summing_to_one():
    with pytest.raises(Exception):
        SplitConfig(train=0.5, val=0.5, test=0.5)
