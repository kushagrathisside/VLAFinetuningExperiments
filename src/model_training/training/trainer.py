"""Orchestrates one experiment's fine-tuning run.

``rde.model_adapters.models.SmolVLAPolicy.fit`` owns the mechanics of one
gradient step (batch construction, forward/backward, optimizer). This module
owns the run around it: periodic checkpointing, periodic validation (on
validation data only, never test), metrics logging, resource
sampling, and the "always checkpoint the final step" guarantee that makes
resume and deployment never depend on rerunning training.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rde.engine import DatasetView
from rde.evaluation import Evaluator
from rde.model_adapters import VisionLanguageAdapter
from rde.model_adapters.models import SmolVLAPolicy

from ..config.schema import ExperimentConfig
from ..tracking.run_dir import RunDir


@dataclass
class TrainingOutcome:
    steps_completed: int
    final_train_loss: float | None
    best_checkpoint: str | None
    best_val_metric: float | None
    selection_metric: str
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    peak_rss_mb: float | None = None
    peak_gpu_mb: float | None = None
    wall_clock_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__}


class Trainer:
    """Fine-tunes one SmolVLA policy against one dataset view."""

    def __init__(self, run_dir: RunDir, config: ExperimentConfig):
        self.run_dir = run_dir
        self.config = config

    def train(
        self,
        train_view: DatasetView,
        val_view: DatasetView,
        *,
        episode_ids: list[int] | None = None,
        episode_weights: dict[int, float] | None = None,
    ) -> TrainingOutcome:
        tc = self.config.training
        ec = self.config.evaluation
        if tc.model != "smolvla":
            raise ValueError(
                f"Trainer implements real gradient-based fine-tuning for model='smolvla' only "
                f"(got {tc.model!r}). Baselines have no training step -- score them via "
                "model_training.evaluation.offline.evaluate_baselines instead."
            )

        try:
            import psutil

            proc: Any = psutil.Process()
        except ImportError:  # pragma: no cover
            proc = None

        checkpoint_dir = self.run_dir.sub("checkpoints")
        policy = SmolVLAPolicy(checkpoint=tc.resume_from or tc.pretrained_checkpoint, device=tc.device)
        # Resuming one of our own checkpoints: it is already configured (shapes
        # and normalization stats) for this exact dataset -- reload as-is. Only
        # a *fresh* pretrained checkpoint (which may target a different
        # embodiment than train_view's) needs its shapes/stats overridden from
        # the dataset contract; see SmolVLAPolicy.load and the cross-dataset
        # generalization notes in docs/EXPERIMENTS.md.
        policy.load(contract=None if tc.resume_from else train_view.contract)
        if tc.resume_from:
            meta = SmolVLAPolicy.verify_checkpoint(tc.resume_from)
            self.run_dir.write_json("resume_source.json", meta)

        import torch  # safe: policy.load() above already required it to succeed

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        peak_rss = proc.memory_info().rss if proc else None

        ids = episode_ids if episode_ids is not None else train_view.episode_ids(split="train")
        if not ids:
            raise ValueError("no training episodes resolved -- nothing to fine-tune on")

        val_adapter = VisionLanguageAdapter(val_view.contract)
        evaluator = Evaluator()
        max_val_frames = tc.val_batches * tc.batch_size

        checkpoints: list[dict[str, Any]] = []
        best: dict[str, Any] = {"metric": None, "checkpoint": None}
        last_step = 0

        def on_step(record: dict[str, Any]) -> None:
            nonlocal peak_rss, last_step
            last_step = record["step"] + 1
            self.run_dir.append_jsonl(
                "metrics/train.jsonl", {"experiment_id": self.config.experiment_id, **record}
            )
            if proc is not None:
                peak_rss = max(peak_rss or 0, proc.memory_info().rss)

            step = last_step  # steps completed so far, 1-indexed
            do_ckpt = bool(tc.checkpoint_every) and step % tc.checkpoint_every == 0
            do_val = bool(tc.validate_every) and step % tc.validate_every == 0
            if not (do_ckpt or do_val):
                return

            val_metric = None
            if do_val:
                policy.eval_mode()
                result = evaluator.evaluate(
                    val_view, policy, val_adapter, split="val", batch_size=tc.batch_size,
                    max_frames=max_val_frames, persist=False,
                )
                policy.train_mode()
                val_metric = result.metrics.get(ec.selection_metric)
                self.run_dir.append_jsonl(
                    "metrics/validation.jsonl",
                    {"experiment_id": self.config.experiment_id, "step": step, **result.metrics},
                )

            if do_ckpt:
                ckpt_path = checkpoint_dir / f"step_{step}"
                policy.save_checkpoint(
                    ckpt_path,
                    extra={
                        "experiment_id": self.config.experiment_id, "step": step,
                        "dataset_id": train_view.dataset_id, "dataset_version": train_view.version,
                        "optimizer_state_saved": False,
                    },
                )
                # A process killed mid-write (disk full, OOM-killer, Ctrl-C)
                # can leave a partial checkpoint directory; verifying before
                # trusting it as `best` (and therefore protected from
                # pruning) keeps a corrupt save from silently displacing a
                # good one.
                try:
                    SmolVLAPolicy.verify_checkpoint(ckpt_path)
                    verified = True
                except (FileNotFoundError, ValueError) as exc:
                    verified = False
                    self.run_dir.append_jsonl(
                        "metrics/checkpoint_errors.jsonl",
                        {"experiment_id": self.config.experiment_id, "step": step,
                         "path": str(ckpt_path), "error": str(exc)},
                    )
                checkpoints.append({
                    "step": step, "path": str(ckpt_path), "val_metric": val_metric,
                    "verified": verified,
                })
                if verified and val_metric is not None:
                    better = (
                        best["metric"] is None
                        or (ec.selection_mode == "min" and val_metric < best["metric"])
                        or (ec.selection_mode == "max" and val_metric > best["metric"])
                    )
                    if better:
                        best.update(metric=val_metric, checkpoint=str(ckpt_path))
                self._prune_checkpoints(checkpoint_dir, checkpoints, tc.keep_last_n_checkpoints, best)

        started = time.perf_counter()
        try:
            fit_result = policy.fit(
                train_view,
                episode_ids=ids,
                episode_weights=episode_weights,
                steps=tc.steps, batch_size=tc.batch_size, learning_rate=tc.learning_rate,
                weight_decay=tc.weight_decay, grad_accum_steps=tc.grad_accum_steps,
                max_grad_norm=tc.max_grad_norm, warmup_steps=tc.warmup_steps,
                seed=tc.seed, on_step=on_step,
            )
        except Exception as exc:
            # A crash here (OOM, a NaN loss, SIGTERM) previously left the run
            # directory in an undetectable half-finished state: partial
            # metrics/checkpoints on disk, no failure marker, and no final
            # checkpoint -- indistinguishable from a run that never started.
            # Record what's known before re-raising, and release the GPU
            # memory this policy was holding rather than leaving it for the
            # caller's next model to contend with.
            self.run_dir.write_json("crash.json", {
                "experiment_id": self.config.experiment_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "steps_completed": last_step,
                "checkpoints": checkpoints,
                "wall_clock_s": time.perf_counter() - started,
            })
            del policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        wall_clock_s = time.perf_counter() - started

        # Always checkpoint the final step, even off-cadence, so resume and
        # deployment never require rerunning training to get a usable artifact.
        final_path = checkpoint_dir / "final"
        policy.save_checkpoint(final_path, extra={
            "experiment_id": self.config.experiment_id, "step": fit_result["steps"],
            "dataset_id": train_view.dataset_id, "dataset_version": train_view.version,
            "final": True,
        })
        checkpoints.append({"step": fit_result["steps"], "path": str(final_path),
                             "val_metric": None, "final": True})

        peak_gpu_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else None
        # The caller (ExperimentRunner) loads a second SmolVLAPolicy for
        # evaluation immediately after this returns; without releasing this
        # one first, both models can be resident on the GPU simultaneously.
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return TrainingOutcome(
            steps_completed=fit_result["steps"],
            final_train_loss=fit_result["final_loss"],
            best_checkpoint=best["checkpoint"] or str(final_path),
            best_val_metric=best["metric"],
            selection_metric=ec.selection_metric,
            checkpoints=checkpoints,
            peak_rss_mb=(peak_rss / 1e6) if peak_rss else None,
            peak_gpu_mb=peak_gpu_mb,
            wall_clock_s=wall_clock_s,
        )

    @staticmethod
    def _prune_checkpoints(
        checkpoint_dir: Path, checkpoints: list[dict[str, Any]], keep_last_n: int, best: dict[str, Any]
    ) -> None:
        """Drop periodic checkpoints beyond ``keep_last_n``, always keeping the
        best-by-validation one regardless of recency."""
        if keep_last_n <= 0:
            return
        import shutil

        periodic = [c for c in checkpoints if not c.get("final")]
        periodic.sort(key=lambda c: c["step"])
        keep_paths = {best.get("checkpoint")} | {c["path"] for c in periodic[-keep_last_n:]}
        for c in periodic[:-keep_last_n] if len(periodic) > keep_last_n else []:
            if c["path"] in keep_paths:
                continue
            p = Path(c["path"])
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
            c["pruned"] = True
