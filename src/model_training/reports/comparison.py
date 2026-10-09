"""Experiment / model x dataset comparison.

Reads each run directory's saved artifacts -- never live training state, so a
comparison is reproducible from disk alone, even for a run from a previous
session.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_run_summary(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"no run directory at {run_dir}")
    summary: dict[str, Any] = {"run_dir": str(run_dir)}
    for name in ("dataset.json", "model.json", "manifest.json", "summary.json"):
        p = run_dir / name
        if p.exists():
            summary[name[:-5]] = json.loads(p.read_text())
    eval_overall = run_dir / "evaluation" / "test_overall.json"
    if eval_overall.exists():
        summary["test_overall"] = json.loads(eval_overall.read_text())
    baselines = run_dir / "evaluation" / "baselines.json"
    if baselines.exists():
        summary["baselines"] = json.loads(baselines.read_text())
    stratified = run_dir / "evaluation" / "stratified.json"
    if stratified.exists():
        summary["stratified"] = json.loads(stratified.read_text())
    return summary


def build_comparison_table(run_dirs: list[str | Path], metric: str = "mse") -> list[dict[str, Any]]:
    """One row per run: model, dataset, intervention, and the requested metric
    both for the model under test and (where present) each baseline."""
    rows: list[dict[str, Any]] = []
    for rd in run_dirs:
        s = load_run_summary(rd)
        dataset = s.get("dataset", {})
        model = s.get("model", {})
        overall = (s.get("test_overall") or {}).get("metrics", {})
        row: dict[str, Any] = {
            "run_dir": s["run_dir"],
            "model": model.get("architecture"),
            "dataset_id": dataset.get("dataset_id"),
            "training_version": dataset.get("training_version"),
            "intervention": dataset.get("kind"),
            metric: overall.get(metric),
        }
        for name, m in (s.get("baselines") or {}).items():
            row[f"{name}_{metric}"] = m.get(metric)
        rows.append(row)
    return rows


def write_comparison_csv(rows: list[dict[str, Any]], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return out_path


def build_model_dataset_matrix(rows: list[dict[str, Any]], metric: str = "mse") -> list[dict[str, Any]]:
    """The model x dataset table: one row per (model, dataset/intervention) pair."""
    return [
        {"model": r.get("model"), "dataset": r.get("training_version"),
         "intervention": r.get("intervention"), metric: r.get(metric)}
        for r in rows
    ]
