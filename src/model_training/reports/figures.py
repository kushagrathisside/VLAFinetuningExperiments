"""Reproducible figures: generated from a run's own metrics files,
never hand-edited. Every function returns the path(s) it wrote, or an empty
list when there is nothing to plot -- never a fabricated placeholder chart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_curves(train_records: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    if not train_records:
        return []
    steps = [r["step"] for r in train_records]
    out: list[Path] = []

    fig, ax = plt.subplots()
    ax.plot(steps, [r["loss"] for r in train_records])
    ax.set_xlabel("step"); ax.set_ylabel("training loss"); ax.set_title("Training loss")
    out.append(_save(fig, Path(out_dir) / "training_loss.png"))

    if any(r.get("lr") is not None for r in train_records):
        fig, ax = plt.subplots()
        ax.plot(steps, [r["lr"] for r in train_records])
        ax.set_xlabel("step"); ax.set_ylabel("learning rate"); ax.set_title("Learning rate schedule")
        out.append(_save(fig, Path(out_dir) / "learning_rate.png"))

    if any(r.get("grad_norm") is not None for r in train_records):
        fig, ax = plt.subplots()
        gn = [r.get("grad_norm") for r in train_records]
        ax.plot([s for s, g in zip(steps, gn) if g is not None], [g for g in gn if g is not None])
        ax.set_xlabel("step"); ax.set_ylabel("gradient norm"); ax.set_title("Gradient norm")
        out.append(_save(fig, Path(out_dir) / "gradient_norm.png"))
    return out


def plot_validation_curve(val_records: list[dict[str, Any]], out_dir: Path, metric: str = "mse") -> Path | None:
    rows = [r for r in val_records if metric in r]
    if not rows:
        return None
    fig, ax = plt.subplots()
    ax.plot([r["step"] for r in rows], [r[metric] for r in rows], marker="o")
    ax.set_xlabel("step"); ax.set_ylabel(f"validation {metric}"); ax.set_title(f"Validation {metric}")
    return _save(fig, Path(out_dir) / f"validation_{metric}.png")


def plot_error_distribution(error_dist: dict[str, Any], out_dir: Path) -> Path | None:
    keys = ["median", "p90", "p95", "p99"]
    if not all(k in error_dist for k in keys):
        return None
    fig, ax = plt.subplots()
    ax.bar(keys, [error_dist[k] for k in keys])
    ax.set_ylabel("mean absolute error"); ax.set_title("Error distribution")
    return _save(fig, Path(out_dir) / "error_distribution.png")


def plot_episode_length_distribution(profile: dict[str, Any], out_dir: Path) -> Path | None:
    hist = (profile.get("episode_length") or {}).get("histogram")
    if not hist:
        return None
    fig, ax = plt.subplots()
    centers = [(b["lo"] + b["hi"]) / 2 for b in hist]
    widths = [b["hi"] - b["lo"] for b in hist]
    ax.bar(centers, [b["count"] for b in hist], width=widths, align="center")
    ax.set_xlabel("episode duration (s)"); ax.set_ylabel("episodes"); ax.set_title("Episode length distribution")
    return _save(fig, Path(out_dir) / "episode_length_distribution.png")


def plot_task_distribution(profile: dict[str, Any], out_dir: Path) -> Path | None:
    tasks = profile.get("tasks") or []
    if not tasks:
        return None
    fig, ax = plt.subplots()
    labels = [f"task_{t['task_id']}" for t in tasks]
    ax.bar(labels, [t["episodes"] for t in tasks])
    ax.set_ylabel("episodes"); ax.set_title("Task distribution")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _save(fig, Path(out_dir) / "task_distribution.png")


def plot_model_comparison(rows: list[dict[str, Any]], metric: str, out_dir: Path,
                            label_key: str = "label") -> Path | None:
    """``rows``: [{label_key: "...", metric: value}, ...] -- one bar per row."""
    rows = [r for r in rows if metric in r and r[metric] is not None]
    if not rows:
        return None
    fig, ax = plt.subplots()
    ax.bar([r[label_key] for r in rows], [r[metric] for r in rows])
    ax.set_ylabel(metric); ax.set_title(f"Model comparison -- {metric}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _save(fig, Path(out_dir) / f"model_comparison_{metric}.png")


def plot_data_fraction_curve(rows: list[dict[str, Any]], metric: str, out_dir: Path) -> Path | None:
    """``rows``: [{"fraction": 0.1, metric: value}, ...] -- the data-efficiency curve."""
    rows = sorted((r for r in rows if metric in r and r[metric] is not None), key=lambda r: r["fraction"])
    if len(rows) < 2:
        return None
    fig, ax = plt.subplots()
    ax.plot([r["fraction"] for r in rows], [r[metric] for r in rows], marker="o")
    ax.set_xlabel("data fraction"); ax.set_ylabel(metric); ax.set_title(f"Data efficiency -- {metric}")
    return _save(fig, Path(out_dir) / f"data_efficiency_{metric}.png")
