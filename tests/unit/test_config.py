from __future__ import annotations

from pathlib import Path

import pytest

from model_training.config import ExperimentConfig, load_experiment_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_baseline_config_loads_and_merges_base():
    cfg = load_experiment_config(CONFIGS / "experiments" / "baseline.yaml")
    assert cfg.dataset.dataset_id == "so100_pickplace"
    assert cfg.dataset.version == "clean/v1"  # inherited from base.yaml
    assert cfg.intervention.kind == "full"
    assert cfg.training.model == "smolvla"  # inherited from base.yaml
    assert cfg.experiment_id.startswith("baseline_full-")


def test_reduced_config_loads_with_fraction():
    cfg = load_experiment_config(CONFIGS / "experiments" / "reduced.yaml")
    assert cfg.intervention.kind == "reduced"
    assert cfg.intervention.fraction == 0.5


def test_imbalanced_template_loads_but_is_empty():
    cfg = load_experiment_config(CONFIGS / "experiments" / "imbalanced.yaml")
    assert cfg.intervention.kind == "custom"
    assert cfg.intervention.episode_ids == []


def test_rebalanced_template_fails_loudly_until_filled_in():
    with pytest.raises(Exception, match="rebalanced"):
        load_experiment_config(CONFIGS / "experiments" / "rebalanced.yaml")


def test_split_fractions_must_sum_to_one():
    with pytest.raises(Exception):
        ExperimentConfig.model_validate(
            {
                "experiment_name": "x",
                "dataset": {"dataset_id": "d", "version": "v"},
                "splits": {"train": 0.5, "val": 0.5, "test": 0.5},
            }
        )


def test_imbalanced_requires_variable_or_episode_ids():
    with pytest.raises(Exception, match="imbalanced"):
        ExperimentConfig.model_validate(
            {
                "experiment_name": "x",
                "dataset": {"dataset_id": "d", "version": "v"},
                "intervention": {"kind": "imbalanced"},
            }
        )


def test_experiment_id_is_deterministic_given_same_inputs():
    data = {
        "experiment_name": "x",
        "dataset": {"dataset_id": "d", "version": "v"},
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    a = ExperimentConfig.model_validate(dict(data))
    b = ExperimentConfig.model_validate(dict(data))
    assert a.experiment_id == b.experiment_id


def test_resolved_dict_contains_full_defaults():
    cfg = load_experiment_config(CONFIGS / "experiments" / "baseline.yaml")
    d = cfg.resolved_dict()
    assert d["training"]["learning_rate"] == pytest.approx(1e-4)
    assert d["evaluation"]["selection_metric"] == "mse"
