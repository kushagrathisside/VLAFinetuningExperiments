from __future__ import annotations

import json

from model_training.config.schema import DatasetRef, EvaluationConfig, ExperimentConfig, SplitConfig, TrainingConfig
from model_training.experiments import ExperimentRunner


def _config(name: str, tmp_root, clean_version: str) -> ExperimentConfig:
    return ExperimentConfig.model_validate({
        "experiment_name": name,
        "hypothesis": "unit test",
        "dataset": {"dataset_id": "so100_test", "version": clean_version},
        "splits": {"train": 0.6, "val": 0.2, "test": 0.2, "seed": 1, "mode": "quantile"},
        # mean_action has no gradient-based fit -- exercises the whole runner
        # (intervention, split integrity, profiling, baselines, report,
        # summary, index) without requiring torch/lerobot.
        "training": {"model": "mean_action", "steps": 1},
        "evaluation": {"split": "test", "baseline_models": ["mean_action", "nearest_neighbor"],
                        "slices": ["per_task"]},
        "tags": {"intervention": "full_data"},
    })


def test_runner_completes_baseline_only_experiment(settings, catalog, clean_version, tmp_path):
    runs_dir = tmp_path / "runs"
    index_dir = tmp_path / "experiments"
    runner = ExperimentRunner(settings=settings, catalog=catalog, runs_dir=runs_dir, index_dir=index_dir)
    cfg = _config("baseline_smoke", tmp_path, clean_version)

    result = runner.run(cfg)

    assert result.run_dir.exists()
    assert (result.run_dir / "dataset.json").exists()
    assert (result.run_dir / "dataset_profile.json").exists()
    assert (result.run_dir / "split_integrity.json").exists()
    assert (result.run_dir / "environment.json").exists()
    assert (result.run_dir / "git.json").exists()
    assert (result.run_dir / "config.resolved.yaml").exists()
    assert (result.run_dir / "evaluation" / "baselines.json").exists()
    assert (result.run_dir / "reports" / "experiment_report.md").exists()
    assert (result.run_dir / "metrics" / "rollout.jsonl").exists()
    assert (result.run_dir / "summary.json").exists()

    split_report = json.loads((result.run_dir / "split_integrity.json").read_text())
    assert split_report["ok"] is True

    baselines = json.loads((result.run_dir / "evaluation" / "baselines.json").read_text())
    assert "mean_action" in baselines and "nearest_neighbor" in baselines

    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["experiment_id"] == cfg.experiment_id
    assert summary["intervention"] == "full"

    report_text = (result.run_dir / "reports" / "experiment_report.md").read_text()
    assert cfg.experiment_name in report_text
    assert "unavailable" in report_text  # rollout status must be stated, not omitted

    # index rows were recorded
    experiments = (index_dir / "experiments.jsonl").read_text().splitlines()
    assert len(experiments) == 1
    datasets = (index_dir / "datasets.jsonl").read_text().splitlines()
    assert len(datasets) == 1


def test_runner_refuses_to_overwrite_an_existing_run(settings, catalog, clean_version, tmp_path):
    runs_dir = tmp_path / "runs"
    index_dir = tmp_path / "experiments"
    runner = ExperimentRunner(settings=settings, catalog=catalog, runs_dir=runs_dir, index_dir=index_dir)
    cfg = _config("no_overwrite", tmp_path, clean_version)
    runner.run(cfg)

    import pytest

    with pytest.raises(FileExistsError):
        runner.run(cfg)


def test_runner_with_reduced_intervention_records_it(settings, catalog, clean_version, tmp_path):
    runs_dir = tmp_path / "runs"
    index_dir = tmp_path / "experiments"
    runner = ExperimentRunner(settings=settings, catalog=catalog, runs_dir=runs_dir, index_dir=index_dir)
    cfg = ExperimentConfig.model_validate({
        "experiment_name": "reduced_smoke",
        "dataset": {"dataset_id": "so100_test", "version": clean_version},
        "splits": {"train": 0.6, "val": 0.2, "test": 0.2, "seed": 1},
        "intervention": {"kind": "reduced", "fraction": 0.5, "seed": 2},
        "training": {"model": "mean_action", "steps": 1},
        "evaluation": {"baseline_models": ["mean_action"], "slices": []},
    })
    result = runner.run(cfg)
    dataset_json = json.loads((result.run_dir / "dataset.json").read_text())
    assert dataset_json["kind"] == "reduced"
    assert len(dataset_json["episode_ids"]) == 4
