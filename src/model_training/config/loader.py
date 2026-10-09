"""YAML loading with base-config merging.

An experiment config is usually ``configs/base.yaml`` overridden by
``configs/experiments/<name>.yaml``. Merging happens before validation, so the
merged, *resolved* result -- not the two source files -- is what
:class:`~model_training.config.schema.ExperimentConfig` validates and what a run
persists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schema import ExperimentConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no config file at {path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def load_experiment_config(
    path: str | Path, base_path: str | Path | None = None
) -> ExperimentConfig:
    """Load one experiment config, optionally merged onto a base config.

    ``base_path`` defaults to ``<same directory's ancestor>/base.yaml`` when a
    sibling ``base.yaml`` exists next to the experiment configs directory --
    matching the ``configs/base.yaml`` + ``configs/experiments/<name>.yaml``
    layout this project ships -- and is otherwise skipped rather than guessed.
    """
    path = Path(path)
    data = load_yaml(path)

    if base_path is None:
        candidate = path.parent.parent / "base.yaml"
        base_path = candidate if candidate.exists() else None

    if base_path is not None:
        base_data = load_yaml(base_path)
        data = _deep_merge(base_data, data)

    data["source_config_path"] = str(path)
    return ExperimentConfig.model_validate(data)
