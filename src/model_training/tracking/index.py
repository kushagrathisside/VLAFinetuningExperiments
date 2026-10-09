"""The lightweight, cross-run experiment index.

Four append-only JSONL files at the project root:

``experiments/experiments.jsonl``   one row per run: identity, dataset, tags, seed
``experiments/metrics.jsonl``        one row per (run, split, metric snapshot)
``experiments/datasets.jsonl``        one row per dataset version an experiment touched
``experiments/models.jsonl``           one row per registered model artifact

Deliberately not a database: every row is one JSON object per line, queryable
with `jq`, `pandas.read_json(lines=True)`, or DuckDB's `read_json_auto` without
this project running -- the "scrapable output" principle applied to the
index itself, not just individual run artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


class JsonlIndex:
    """Append-only JSONL file with a simple in-Python query helper."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def query(self, **filters: Any) -> Iterator[dict[str, Any]]:
        for row in self.read_all():
            if all(row.get(k) == v for k, v in filters.items()):
                yield row


class ExperimentIndex:
    """The four JSONL files, rooted at ``<project>/experiments/``."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.experiments = JsonlIndex(self.root / "experiments.jsonl")
        self.metrics = JsonlIndex(self.root / "metrics.jsonl")
        self.datasets = JsonlIndex(self.root / "datasets.jsonl")
        self.models = JsonlIndex(self.root / "models.jsonl")

    def record_experiment(self, **fields: Any) -> None:
        self.experiments.append(fields)

    def record_metric(self, **fields: Any) -> None:
        self.metrics.append(fields)

    def record_dataset(self, **fields: Any) -> None:
        self.datasets.append(fields)

    def record_model(self, **fields: Any) -> None:
        self.models.append(fields)
