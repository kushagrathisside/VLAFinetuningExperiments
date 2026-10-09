"""Model artifact identity: every field needed to answer "what
exactly is this checkpoint" without re-deriving it from the run directory."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any

from rde.model_adapters.models import SmolVLAPolicy


@dataclass
class ModelArtifact:
    model_id: str
    model_version: str
    artifact_path: str
    architecture: str
    pretrained_model: str
    training_run_id: str
    experiment_id: str
    dataset_id: str
    dataset_version: str
    observation_schema: dict[str, Any]
    action_schema: dict[str, Any]
    embodiment: str | None
    dependencies: dict[str, Any]
    git_commit: dict[str, Any]
    created_at: str
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__}


def build_model_artifact(
    *,
    checkpoint_path: str,
    contract: dict[str, Any],
    run_dir_path: str,
    experiment_id: str,
    tags: dict[str, str],
    dataset_id: str,
    dataset_version: str,
    environment: dict[str, Any],
    git_info: dict[str, Any],
    metrics: dict[str, Any],
) -> ModelArtifact:
    """Assemble a :class:`ModelArtifact` from a checkpoint already saved by
    ``SmolVLAPolicy.save_checkpoint`` -- verified, not assumed complete."""
    meta = SmolVLAPolicy.verify_checkpoint(checkpoint_path)
    model_id = f"{tags.get('model', 'smolvla')}-{experiment_id}"
    return ModelArtifact(
        model_id=model_id,
        model_version=str(meta.get("step", "final")),
        artifact_path=str(checkpoint_path),
        architecture="smolvla",
        pretrained_model=meta.get("checkpoint_source", ""),
        training_run_id=str(run_dir_path),
        experiment_id=experiment_id,
        dataset_id=meta.get("dataset_id", dataset_id),
        dataset_version=meta.get("dataset_version", dataset_version),
        observation_schema=contract.get("observation", {}),
        action_schema=contract.get("action", {}),
        embodiment=tags.get("embodiment"),
        dependencies={"torch": environment.get("torch"), "lerobot": environment.get("lerobot")},
        git_commit=git_info,
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        metrics=metrics,
    )
