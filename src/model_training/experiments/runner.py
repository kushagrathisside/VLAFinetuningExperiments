"""ExperimentRunner: the full lifecycle, from an
:class:`~model_training.config.schema.ExperimentConfig` to a completed run
directory with metrics, checkpoints, evaluation, failure analysis, a feedback
artifact and reports.

Every step below is either a call into ``rde`` (dataset resolution,
intervention construction, training, evaluation, failure detection) or
packaging of that call's real output -- nothing here fabricates a result. A
step that cannot run (no GPU, no simulator) records why in the run directory
rather than being silently skipped.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from rde.catalog import Catalog
from rde.config import Settings, get_settings
from rde.engine import DataEngine

from ..config.schema import ExperimentConfig
from ..data.interventions import build_intervention_dataset
from ..data.profile import build_dataset_profile
from ..data.split_integrity import verify_split_integrity
from ..deployment.artifact import build_model_artifact
from ..evaluation.offline import episode_metrics, error_distribution, evaluate_baselines, stratified_evaluate, temporal_error
from ..feedback.artifact import build_feedback_artifact
from ..reports import figures
from ..reports.experiment_report import render_experiment_report
from ..reports.model_card import render_model_card
from ..tracking.index import ExperimentIndex
from ..tracking.run_dir import RunDir, collect_environment, collect_git_info, make_run_dir


def _default_repo_root() -> Path:
    # .../vla-finetune/src/model_training/experiments/runner.py -> .../VLAExperiments
    return Path(__file__).resolve().parents[4]


RDE_REPO = Path(os.environ.get("RDE_REPO_PATH", str(_default_repo_root() / "episodic-data-engine")))
MT_REPO = Path(os.environ.get("MODEL_TRAINING_REPO_PATH", str(Path(__file__).resolve().parents[3])))


@dataclass
class ExperimentRunResult:
    run_dir: Path
    experiment_id: str
    training_version: str
    summary: dict[str, Any]


class ExperimentRunner:
    def __init__(
        self,
        settings: Settings | None = None,
        catalog: Catalog | None = None,
        runs_dir: Path | None = None,
        index_dir: Path | None = None,
    ):
        self.settings = settings or get_settings()
        self.catalog = catalog or Catalog(self.settings.paths.catalog_db)
        self.runs_dir = Path(runs_dir) if runs_dir else MT_REPO / "runs"
        self.index = ExperimentIndex(Path(index_dir) if index_dir else MT_REPO / "experiments")

    # -- entry point ----------------------------------------------------------
    def run(self, config: ExperimentConfig, seed: int | None = None) -> ExperimentRunResult:
        seed = seed if seed is not None else config.seeds[0]
        run_dir = make_run_dir(self.runs_dir, config.experiment_name, seed)

        environment = collect_environment()
        git_info = collect_git_info(RDE_REPO, MT_REPO)
        run_dir.write_json("environment.json", environment)
        run_dir.write_json("git.json", git_info)
        (run_dir.path / "config.resolved.yaml").write_text(
            yaml.safe_dump(config.resolved_dict(), sort_keys=False)
        )

        limitations: list[str] = []
        if environment.get("gpu") is None:
            limitations.append("no GPU detected in this environment at run time -- training ran on CPU if it ran at all")
        if environment.get("torch") is None:
            limitations.append("torch not installed in this environment -- this run could not perform real fine-tuning")

        # -- 1. dataset intervention: a real, lineage-tracked version --------
        intervention_result = build_intervention_dataset(
            config.dataset.dataset_id, config.dataset.version, config.intervention,
            config.splits, settings=self.settings, catalog=self.catalog,
        )
        run_dir.write_json("dataset.json", {
            "dataset_id": config.dataset.dataset_id,
            "input_version": config.dataset.version,
            "clean_version": intervention_result.clean_version,
            "training_version": intervention_result.training_version,
            "kind": intervention_result.kind,
            "episode_ids": intervention_result.episode_ids,
            "excluded_episode_ids": intervention_result.excluded_episode_ids,
            "episode_weights": intervention_result.episode_weights,
            "selection_reason": intervention_result.selection_reason,
            "build_summary": intervention_result.build_summary,
        })
        self.index.record_dataset(
            experiment_id=config.experiment_id, dataset_id=config.dataset.dataset_id,
            training_version=intervention_result.training_version, kind=intervention_result.kind,
            episode_count=len(intervention_result.episode_ids),
            recorded_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        )

        training_version = intervention_result.training_version
        engine = DataEngine(self.settings)
        view = engine.get_dataset(config.dataset.dataset_id, training_version)
        try:
            split_report = verify_split_integrity(view)  # raises on leakage
            run_dir.write_json("split_integrity.json", split_report.to_dict())

            profile = build_dataset_profile(
                config.dataset.dataset_id, training_version, settings=self.settings, catalog=self.catalog,
            )
            run_dir.write_json("dataset_profile.json", profile.to_dict())
            figures.plot_task_distribution(profile.to_dict(), run_dir.sub("reports") / "figures")
            figures.plot_episode_length_distribution(profile.to_dict(), run_dir.sub("reports") / "figures")

            # -- 2. training -----------------------------------------------------
            train_ids = view.episode_ids(split="train")
            episode_weights = {
                eid: float((view.episode(eid).attributes or {}).get("sampling_weight", 1.0))
                for eid in train_ids
            }
            episode_weights = episode_weights if any(w != 1.0 for w in episode_weights.values()) else None

            training_outcome = None
            selected_checkpoint = None
            if config.training.model == "smolvla":
                from ..training.trainer import Trainer

                trainer = Trainer(run_dir, config)
                training_outcome = trainer.train(
                    view, view, episode_ids=train_ids, episode_weights=episode_weights,
                )
                run_dir.write_json("training_outcome.json", training_outcome.to_dict())
                figures.plot_training_curves(
                    [json.loads(l) for l in (run_dir.path / "metrics" / "train.jsonl").read_text().splitlines()]
                    if (run_dir.path / "metrics" / "train.jsonl").exists() else [],
                    run_dir.sub("reports") / "figures",
                )
                val_path = run_dir.path / "metrics" / "validation.jsonl"
                if val_path.exists():
                    val_records = [json.loads(l) for l in val_path.read_text().splitlines()]
                    figures.plot_validation_curve(val_records, run_dir.sub("reports") / "figures",
                                                    metric=config.evaluation.selection_metric)
                selected_checkpoint = training_outcome.best_checkpoint
            else:
                limitations.append(
                    f"training.model={config.training.model!r} has no gradient-based fit; "
                    "this run only scored baselines, no fine-tuning occurred"
                )

            run_dir.write_json("model.json", {
                "architecture": config.training.model,
                "pretrained_checkpoint": config.training.pretrained_checkpoint,
                "selected_checkpoint": selected_checkpoint,
                "selection_metric": config.evaluation.selection_metric,
                "selection_value": training_outcome.best_val_metric if training_outcome else None,
            })

            # -- 3. offline test evaluation ---------------------------------------
            baseline_results = evaluate_baselines(
                view, config.evaluation.baseline_models, split=config.evaluation.split,
                batch_size=config.evaluation.batch_size, max_frames=config.evaluation.max_frames,
            )
            eval_dir = run_dir.sub("evaluation")
            eval_dir_path = eval_dir
            baseline_json = {name: r.metrics for name, r in baseline_results.items()}
            (eval_dir_path / "baselines.json").write_text(json.dumps(baseline_json, indent=2, default=str))

            test_overall = None
            stratified = None
            feedback_artifact = None
            if selected_checkpoint:
                from rde.model_adapters.models import SmolVLAPolicy

                policy = SmolVLAPolicy.from_checkpoint(selected_checkpoint)
                from rde.model_adapters import VisionLanguageAdapter

                adapter = VisionLanguageAdapter(view.contract)
                strat_results = stratified_evaluate(
                    view, policy, adapter, base_split=config.evaluation.split,
                    batch_size=config.evaluation.batch_size, max_frames=config.evaluation.max_frames,
                    slices=config.evaluation.slices,
                )
                test_overall = strat_results.pop("overall")
                stratified = {name: {**r.metrics, "frames": r.frames_evaluated} for name, r in strat_results.items()}

                overall_payload = {
                    "summary": test_overall.summary(),
                    "metrics": test_overall.metrics,
                    "error_distribution": error_distribution(test_overall),
                    "episode_metrics": episode_metrics(test_overall),
                    "temporal_error": temporal_error(test_overall, view),
                }
                (eval_dir_path / "test_overall.json").write_text(json.dumps(overall_payload, indent=2, default=str))
                (eval_dir_path / "stratified.json").write_text(json.dumps(stratified, indent=2, default=str))
                figures.plot_error_distribution(overall_payload["error_distribution"], run_dir.sub("reports") / "figures")

                predictions_dir = run_dir.sub("predictions")
                with (predictions_dir / "test_outcomes.jsonl").open("w") as f:
                    for o in test_overall.outcomes:
                        f.write(json.dumps({
                            "run_id": str(run_dir.path), "experiment_id": config.experiment_id,
                            "dataset_id": config.dataset.dataset_id, "dataset_version": training_version,
                            "frame_id": o.frame_id, "episode_id": o.episode_id,
                            "error": o.error, "confidence": o.confidence, "per_dim_error": o.per_dim_error,
                        }, default=str) + "\n")

                model_artifact = build_model_artifact(
                    checkpoint_path=selected_checkpoint, contract=view.contract,
                    run_dir_path=str(run_dir.path), experiment_id=config.experiment_id, tags=config.tags,
                    dataset_id=config.dataset.dataset_id, dataset_version=training_version,
                    environment=environment, git_info=git_info, metrics=test_overall.metrics,
                )
                self.index.record_model(**model_artifact.to_dict())

                feedback_artifact = build_feedback_artifact(
                    config.experiment_id, model_artifact.model_id, config.dataset.dataset_id, training_version,
                    test_overall, settings=self.settings, catalog=self.catalog,
                )
                (run_dir.sub("failures") / "feedback_artifact.json").write_text(
                    json.dumps(feedback_artifact, indent=2, default=str)
                )

                run_dir.write_json("model_artifact.json", model_artifact.to_dict())
                (run_dir.sub("reports") / "model_card.md").write_text(
                    render_model_card(model_artifact.to_dict(), {
                        "training_config": config.training.model_dump(),
                        "failure_notes": feedback_artifact["candidate_explanations"],
                        "limitations": limitations,
                    })
                )

            # -- 4. metrics.jsonl -> test.jsonl / rollout.jsonl -------------------
            if test_overall is not None:
                run_dir.append_jsonl("metrics/test.jsonl", {
                    "experiment_id": config.experiment_id, **test_overall.metrics,
                })
            run_dir.append_jsonl("metrics/rollout.jsonl", {
                "experiment_id": config.experiment_id, "status": "unavailable",
                "reason": "no simulator or physical robot connected in this environment",
            })

            # -- 5. report ---------------------------------------------------------
            report_ctx = {
                "config": config.resolved_dict(),
                "run_dir": str(run_dir.path),
                "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "dataset": {
                    "dataset_id": config.dataset.dataset_id, "input_version": config.dataset.version,
                    "clean_version": intervention_result.clean_version,
                    "training_version": training_version,
                    "selection_reason": intervention_result.selection_reason,
                    "episode_ids": intervention_result.episode_ids,
                    "excluded_episode_ids": intervention_result.excluded_episode_ids,
                },
                "profile": profile.to_dict(),
                "model": {
                    "architecture": config.training.model,
                    "selected_checkpoint": selected_checkpoint,
                    "selection_value": training_outcome.best_val_metric if training_outcome else None,
                },
                "training_outcome": training_outcome.to_dict() if training_outcome else None,
                "environment": {**environment, "git": git_info},
                "test_metrics": {"overall": test_overall.metrics} if test_overall else None,
                "baseline_metrics": baseline_json,
                "model_test_mse": test_overall.metrics.get("mse") if test_overall else None,
                "stratified_metrics": stratified,
                "feedback_artifact": feedback_artifact,
                "rollout_status": "unavailable -- no simulator or robot connected in this environment",
                "limitations": limitations,
                "conclusion": None,
            }
            report_md = render_experiment_report(report_ctx)
            (run_dir.sub("reports") / "experiment_report.md").write_text(report_md)

            summary = {
                "experiment_id": config.experiment_id, "experiment_name": config.experiment_name,
                "dataset_id": config.dataset.dataset_id, "training_version": training_version,
                "intervention": intervention_result.kind, "tags": config.tags,
                "seed": seed, "model": config.training.model,
                "test_mse": test_overall.metrics.get("mse") if test_overall else None,
                "baseline_mean_action_mse": baseline_json.get("mean_action", {}).get("mse"),
                "steps_completed": training_outcome.steps_completed if training_outcome else None,
                "wall_clock_s": training_outcome.wall_clock_s if training_outcome else None,
            }
            run_dir.write_json("summary.json", summary)
            self.index.record_experiment(
                run_dir=str(run_dir.path), created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
                **summary,
            )
            if test_overall is not None:
                self.index.record_metric(
                    experiment_id=config.experiment_id, run_dir=str(run_dir.path),
                    split=config.evaluation.split, **test_overall.metrics,
                )
        finally:
            view.close()

        return ExperimentRunResult(
            run_dir=run_dir.path, experiment_id=config.experiment_id,
            training_version=training_version, summary=summary,
        )
