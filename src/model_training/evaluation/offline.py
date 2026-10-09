"""Offline evaluation: baselines, error distributions, episode/temporal
metrics and stratified slices -- built entirely on top of
``rde.evaluation.Evaluator``, not a replacement
for it.

Never converts an offline prediction-accuracy number into a claimed robot
success rate: ``success_rate`` here is exactly Evaluator's own tolerance-band
proxy, already labelled as a proxy in ``rde.evaluation.evaluator``'s own
docstring, carried through unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rde.engine import DatasetView
from rde.evaluation import Evaluator
from rde.evaluation.evaluator import EvaluationResult
from rde.model_adapters import MODEL_REGISTRY, build_model
from rde.model_adapters.base import Model, ModelInputAdapter


def evaluate_baselines(
    view: DatasetView,
    names: list[str],
    *,
    split: str = "test",
    train_split: str = "train",
    batch_size: int = 32,
    max_frames: int | None = None,
) -> dict[str, EvaluationResult]:
    """Fit and score each baseline through the exact call used for the model
    under test, so the comparison is apples-to-apples."""
    ev = Evaluator()
    out: dict[str, EvaluationResult] = {}
    for name in names:
        if name not in MODEL_REGISTRY:
            raise KeyError(f"unknown baseline model {name!r}; known: {sorted(MODEL_REGISTRY)}")
        model, adapter = build_model(name, view.contract)
        model.fit(view, split=train_split)
        out[name] = ev.evaluate(
            view, model, adapter, split=split, batch_size=batch_size,
            max_frames=max_frames, persist=False,
        )
    return out


def error_distribution(result: EvaluationResult) -> dict[str, Any]:
    """Median / p90 / p95 / p99 of per-frame mean-absolute error."""
    if not result.outcomes:
        return {"status": "no outcomes"}
    errs = np.array([o.error for o in result.outcomes])
    return {
        "median": float(np.median(errs)),
        "p90": float(np.percentile(errs, 90)),
        "p95": float(np.percentile(errs, 95)),
        "p99": float(np.percentile(errs, 99)),
        "min": float(errs.min()),
        "max": float(errs.max()),
    }


def episode_metrics(result: EvaluationResult) -> list[dict[str, Any]]:
    """Per-episode MSE / MAE / RMSE / max error."""
    by_ep: dict[int, list[float]] = {}
    for o in result.outcomes:
        by_ep.setdefault(o.episode_id, []).append(o.error)
    out = []
    for ep, errs in sorted(by_ep.items()):
        a = np.asarray(errs)
        out.append({
            "episode_id": ep, "frames": len(a), "mae": float(a.mean()),
            "mse": float((a ** 2).mean()), "rmse": float(np.sqrt((a ** 2).mean())),
            "max_error": float(a.max()),
        })
    return out


def temporal_error(result: EvaluationResult, view: DatasetView, buckets: int = 10) -> dict[str, Any]:
    """Mean error vs. normalized episode progress.

    ``FrameOutcome`` carries ``frame_id``/``episode_id`` but not
    ``frame_index``, so this re-reads (once per distinct episode already
    touched by the evaluation, not the whole dataset) the small frame_id ->
    frame_index mapping via ``DatasetView.get_episode_frames`` -- bounded by
    episode length, the same pattern the rest of the engine uses.
    """
    if not result.outcomes:
        return {"status": "no outcomes"}
    episodes_touched = {o.episode_id for o in result.outcomes}
    frame_index: dict[int, int] = {}
    ep_len: dict[int, int] = {}
    for ep_id in episodes_touched:
        table = view.get_episode_frames(ep_id, columns=["frame_id", "frame_index"])
        d = table.to_pydict()
        ep_len[ep_id] = max(1, len(d["frame_id"]))
        for fid, fidx in zip(d["frame_id"], d["frame_index"]):
            frame_index[int(fid)] = int(fidx)

    bucket_errors: list[list[float]] = [[] for _ in range(buckets)]
    unresolved = 0
    for o in result.outcomes:
        fidx = frame_index.get(o.frame_id)
        if fidx is None:
            unresolved += 1
            continue
        length = ep_len[o.episode_id]
        progress = fidx / (length - 1) if length > 1 else 0.0
        b = min(buckets - 1, max(0, int(progress * buckets)))
        bucket_errors[b].append(o.error)

    return {
        "buckets": buckets,
        "progress_bucket_edges": [round(i / buckets, 2) for i in range(buckets + 1)],
        "mean_error_by_progress_bucket": [
            round(float(np.mean(b)), 6) if b else None for b in bucket_errors
        ],
        "n_by_bucket": [len(b) for b in bucket_errors],
        "unresolved_frames": unresolved,
    }


def stratified_evaluate(
    view: DatasetView,
    model: Model,
    adapter: ModelInputAdapter,
    *,
    base_split: str = "test",
    batch_size: int = 32,
    max_frames: int | None = None,
    slices: list[str] | None = None,
) -> dict[str, EvaluationResult]:
    """Evaluate ``overall`` plus each requested named slice against the same
    base_split episode pool (fixed evaluation slices).

    Slice names are resolved from real per-episode metadata, never invented:

    ``per_task``            one slice per distinct ``task_id`` present.
    ``per_length_bucket``    short/medium/long by within-split episode-length
                             terciles.

    Anything requiring metadata this dataset's schema does not carry (an
    explicit "variation" field, OOD labels) is not silently approximated here
    -- see ``model_training.data.profile``'s coverage section for what is and
    is not computable, and ``analyze_failures`` for the *model-driven*
    "hard_cases" slice, which is a different kind of thing (evidence about a
    specific model, not a property of the dataset alone).
    """
    ev = Evaluator()
    out: dict[str, EvaluationResult] = {
        "overall": ev.evaluate(view, model, adapter, split=base_split, batch_size=batch_size,
                                max_frames=max_frames, persist=False)
    }
    for name in slices or []:
        if name == "per_task":
            for task_id in sorted({e.task_id for e in view.episodes if e.task_id is not None}):
                ep_ids = view.episode_ids(split=base_split, task_id=task_id)
                if not ep_ids:
                    continue
                out[f"task_{task_id}"] = ev.evaluate(
                    view, model, adapter, split=f"{base_split}:task_{task_id}",
                    episode_ids=ep_ids, batch_size=batch_size, max_frames=max_frames, persist=False,
                )
        elif name == "per_length_bucket":
            lengths = {
                e.episode_id: e.num_frames for e in view.episodes
                if e.attributes.get("split") == base_split
            }
            if len(lengths) >= 3:
                t1, t2 = np.percentile(list(lengths.values()), [33.3, 66.7])
                groups: dict[str, list[int]] = {"short": [], "medium": [], "long": []}
                for eid, n in lengths.items():
                    groups["short" if n <= t1 else "long" if n > t2 else "medium"].append(eid)
                for bucket_name, ids in groups.items():
                    if ids:
                        out[f"length_{bucket_name}"] = ev.evaluate(
                            view, model, adapter, split=f"{base_split}:length_{bucket_name}",
                            episode_ids=ids, batch_size=batch_size, max_frames=max_frames, persist=False,
                        )
        else:
            raise ValueError(
                f"unknown evaluation slice {name!r}; supported: 'per_task', 'per_length_bucket' "
                "(a model-driven 'hard_cases' slice comes from analyze_failures, not here)"
            )
    return out
