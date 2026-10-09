#!/usr/bin/env python
"""Render the README results figure from completed run directories.

Reads only artifacts every run writes (``metrics/validation.jsonl``,
``evaluation/test_overall.json``, ``evaluation/baselines.json``), so the figure
is reproducible from disk:

    python scripts/plot_results.py \
        --so100 runs/<date>/baseline_full/seed_0 \
        --so100-continued runs/<date>/baseline_full_extended/seed_0 \
        --aloha runs/<date>/aloha_coffee_baseline/seed_0 \
        --out docs/figures/results.svg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

SMOLVLA, NEIGHBOR, MEAN = "#2a78d6", "#eb6834", "#8a8a85"
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e6e6e3"


def _validation(run: Path) -> list[tuple[int, float]]:
    rows = [json.loads(line) for line in (run / "metrics/validation.jsonl").read_text().splitlines() if line]
    return [(int(r["step"]), float(r["mse"])) for r in rows]


def _test(run: Path) -> tuple[float, dict[str, float]]:
    overall = json.loads((run / "evaluation/test_overall.json").read_text())
    smolvla = float(overall["summary"]["mse"])
    baselines = json.loads((run / "evaluation/baselines.json").read_text())
    return smolvla, {k: float(v["mse"]) for k, v in baselines.items()}


def _style(ax, title: str) -> None:
    if ax.get_yscale() == "log":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c9c4")
    ax.tick_params(colors=MUTED, labelsize=9)


def _curve(ax, points, selected_step: int) -> None:
    steps, mse = zip(*points)
    ax.plot(steps, mse, color=SMOLVLA, linewidth=2, marker="o", markersize=5)
    sel = dict(points)[selected_step]
    ax.plot([selected_step], [sel], marker="o", markersize=11, markerfacecolor="none",
            markeredgecolor=INK, markeredgewidth=1.4)
    ax.annotate(f"selected checkpoint\nMSE {sel:.4f}", (selected_step, sel), textcoords="offset points",
                xytext=(0, -12), ha="center", va="top", fontsize=8.5, color=INK)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so100", type=Path, required=True)
    ap.add_argument("--so100-continued", type=Path, required=True)
    ap.add_argument("--aloha", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path"})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), gridspec_kw={"width_ratios": [1.25, 1, 1.15]})
    fig.patch.set_facecolor("white")

    # (a) SO-100: first 1,000 steps, then the continuation from step 1,000.
    first = _validation(a.so100)
    cont = [(1000 + s, m) for s, m in _validation(a.so100_continued)]
    ax = axes[0]
    _curve(ax, first + cont, selected_step=1800)
    ax.axvline(1000, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(1030, 0.12, "resumed,\nlr 1e-4 → 3e-5", fontsize=8.5, color=MUTED, va="top")
    ax.set_yscale("log")
    ax.set_ylim(6e-4, 0.3)
    ax.set_xlabel("optimization step", fontsize=9, color=MUTED)
    ax.set_ylabel("validation MSE (log scale)", fontsize=9, color=MUTED)
    _style(ax, "(a) SO-100 validation MSE")

    # (b) ALOHA.
    ax = axes[1]
    _curve(ax, _validation(a.aloha), selected_step=800)
    ax.set_yscale("log")
    ax.set_ylim(0.09, 1.6)
    ax.set_yticks([0.1, 0.2, 0.5, 1.0])
    ax.set_xlabel("optimization step", fontsize=9, color=MUTED)
    _style(ax, "(b) ALOHA coffee validation MSE")

    # (c) Test MSE against both baselines.
    ax = axes[2]
    so_smol, so_base = _test(a.so100_continued)
    al_smol, al_base = _test(a.aloha)
    groups = [("SO-100", so_base["mean_action"], so_base["nearest_neighbor"], so_smol),
              ("ALOHA coffee", al_base["mean_action"], al_base["nearest_neighbor"], al_smol)]
    width = 0.26
    series = [("mean action", MEAN, 1), ("nearest neighbour", NEIGHBOR, 2), ("SmolVLA", SMOLVLA, 3)]
    for gi, g in enumerate(groups):
        for si, (label, color, idx) in enumerate(series):
            x = gi + (si - 1) * (width + 0.02)
            ax.bar(x, g[idx], width=width, color=color, label=label if gi == 0 else None)
            ax.text(x, g[idx] * 1.12, f"{g[idx]:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks([0, 1], [g[0] for g in groups])
    ax.set_yscale("log")
    ax.set_ylim(0.01, 3)
    ax.set_ylabel("test MSE (log scale)", fontsize=9, color=MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3, handlelength=1, columnspacing=1)
    _style(ax, "(c) Test MSE vs. baselines")

    fig.tight_layout(w_pad=2.5)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
