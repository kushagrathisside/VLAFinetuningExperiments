"""A lightweight local model registry -- built on the same JSONL
mechanism as the rest of the tracking index, not a second storage layer.

Answers, directly from one file:

    Which model is this?              -> get(model_id)
    Which dataset trained it?          -> row["dataset_id"] / ["dataset_version"]
    Which experiment created it?        -> row["experiment_id"]
    Which checkpoint is deployable?       -> row["artifact_path"]
    What metrics justified selection?      -> row["metrics"]
    Which code version produced it?         -> row["git_commit"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..tracking.index import JsonlIndex
from .artifact import ModelArtifact


class ModelRegistry:
    def __init__(self, path: Path):
        self._index = JsonlIndex(Path(path))

    def register(self, artifact: ModelArtifact) -> None:
        self._index.append(artifact.to_dict())

    def get(self, model_id: str) -> dict[str, Any] | None:
        """The most recent record for ``model_id``.

        The backing store is append-only, so a re-registered model_id (e.g.
        metrics recomputed for the same artifact) means more than one row
        can match; returning the first match returned the oldest, stalest
        one. Later rows supersede earlier ones for the same id.
        """
        row = None
        for row in self._index.query(model_id=model_id):
            pass
        return row

    def list(self) -> list[dict[str, Any]]:
        return self._index.read_all()

    def for_experiment(self, experiment_id: str) -> list[dict[str, Any]]:
        return list(self._index.query(experiment_id=experiment_id))
