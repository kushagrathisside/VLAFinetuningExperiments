"""``mtrain`` -- the model_training CLI.

Console script name is ``mtrain``, not a bare ``model``, to avoid
colliding with the engine's own ``rde model ...`` sub-app and every other
program named ``model`` a shell might resolve first.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


def _print_json(data) -> None:
    console.print_json(json.dumps(data, indent=2, default=str))


@app.command()
def inspect(
    dataset: str = typer.Option(..., "--dataset"),
    version: str = typer.Option("training/v1", "--version"),
):
    """Dataset overview -- a thin passthrough to rde's own explorer."""
    from rde.catalog import Catalog
    from rde.config import get_settings
    from rde.exploration.queries import DatasetExplorer

    settings = get_settings()
    explorer = DatasetExplorer(settings=settings, catalog=Catalog(settings.paths.catalog_db))
    _print_json(explorer.overview(dataset, version).to_dict())


@app.command()
def profile(
    dataset: str = typer.Option(..., "--dataset"),
    version: str = typer.Option(..., "--version"),
    out: Optional[Path] = typer.Option(None, "--out"),
):
    """Build and print a dataset profile; optionally save it."""
    from model_training.data import build_dataset_profile

    p = build_dataset_profile(dataset, version)
    d = p.to_dict()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(d, indent=2, default=str))
        console.print(f"wrote {out}")
    _print_json(d)


@app.command()
def train(config: Path = typer.Option(..., "--config"), seed: Optional[int] = typer.Option(None, "--seed")):
    """Run one experiment end to end: intervention -> train -> evaluate -> report."""
    from model_training.config import load_experiment_config
    from model_training.experiments import ExperimentRunner

    cfg = load_experiment_config(config)
    result = ExperimentRunner().run(cfg, seed=seed)
    console.print(f"run complete: {result.run_dir}")
    _print_json(result.summary)


@app.command()
def validate(run: Path = typer.Option(..., "--run")):
    """Report the selected checkpoint's validation metrics for a completed run."""
    outcome_path = run / "training_outcome.json"
    if not outcome_path.exists():
        console.print(f"[yellow]no training_outcome.json in {run} -- this run had no gradient-based training[/yellow]")
        raise typer.Exit(1)
    outcome = json.loads(outcome_path.read_text())
    val_path = run / "metrics" / "validation.jsonl"
    history = [json.loads(l) for l in val_path.read_text().splitlines()] if val_path.exists() else []
    _print_json({
        "best_checkpoint": outcome["best_checkpoint"], "selection_metric": outcome["selection_metric"],
        "best_val_metric": outcome["best_val_metric"], "validation_history": history,
    })


@app.command()
def evaluate(
    run: Path = typer.Option(..., "--run"),
    split: str = typer.Option("test", "--split"),
):
    """Re-run stratified + baseline offline evaluation for an existing run's
    selected checkpoint against SPLIT, overwriting its evaluation/ artifacts."""
    from rde.catalog import Catalog
    from rde.config import get_settings
    from rde.engine import DataEngine
    from rde.model_adapters import VisionLanguageAdapter
    from rde.model_adapters.models import SmolVLAPolicy

    from model_training.evaluation import episode_metrics, error_distribution, evaluate_baselines, stratified_evaluate, temporal_error

    dataset_json = json.loads((run / "dataset.json").read_text())
    model_json = json.loads((run / "model.json").read_text())
    checkpoint = model_json.get("selected_checkpoint")
    if not checkpoint:
        console.print(f"[yellow]no selected_checkpoint recorded for run {run}[/yellow]")
        raise typer.Exit(1)

    settings = get_settings()
    view = DataEngine(settings).get_dataset(dataset_json["dataset_id"], dataset_json["training_version"])
    try:
        policy = SmolVLAPolicy.from_checkpoint(checkpoint)
        adapter = VisionLanguageAdapter(view.contract)
        strat = stratified_evaluate(view, policy, adapter, base_split=split, slices=["per_task"])
        overall = strat.pop("overall")
        payload = {
            "summary": overall.summary(), "metrics": overall.metrics,
            "error_distribution": error_distribution(overall),
            "episode_metrics": episode_metrics(overall),
            "temporal_error": temporal_error(overall, view),
        }
        eval_dir = run / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        (eval_dir / "test_overall.json").write_text(json.dumps(payload, indent=2, default=str))
        (eval_dir / "stratified.json").write_text(
            json.dumps({k: {**r.metrics, "frames": r.frames_evaluated} for k, r in strat.items()}, indent=2, default=str)
        )
        _print_json(payload["summary"])
    finally:
        view.close()


@app.command()
def test(run: Path = typer.Option(..., "--run")):
    """Alias for `evaluate --split test` -- the offline test-set score."""
    evaluate(run=run, split="test")


@app.command()
def rollout(run: Path = typer.Option(..., "--run")):
    """Closed-loop rollout evaluation. No simulator or robot is connected in
    this environment, so this reports 'unavailable' rather than fabricating a
    success rate -- never conflated with offline evaluation."""
    payload = {"status": "unavailable", "reason": "no simulator or physical robot connected in this environment"}
    (run / "metrics").mkdir(exist_ok=True)
    with (run / "metrics" / "rollout.jsonl").open("a") as f:
        f.write(json.dumps(payload) + "\n")
    _print_json(payload)


@app.command()
def compare(
    runs: list[Path] = typer.Option(..., "--runs"),
    metric: str = typer.Option("mse", "--metric"),
    out: Optional[Path] = typer.Option(None, "--out"),
):
    """Model x dataset comparison table across several completed runs."""
    from model_training.reports import build_comparison_table, write_comparison_csv

    rows = build_comparison_table(runs, metric=metric)
    table = Table(*rows[0].keys()) if rows else Table("run_dir")
    for r in rows:
        table.add_row(*[str(v) for v in r.values()])
    console.print(table)
    if out:
        write_comparison_csv(rows, out)
        console.print(f"wrote {out}")


@app.command("analyze-failures")
def analyze_failures(run: Path = typer.Option(..., "--run")):
    """Print the feedback artifact (ranked failures, candidate explanations,
    recommended next action) for a completed run."""
    path = run / "failures" / "feedback_artifact.json"
    if not path.exists():
        console.print(f"[yellow]no feedback_artifact.json in {run} -- this run had no test-set model evaluation[/yellow]")
        raise typer.Exit(1)
    _print_json(json.loads(path.read_text()))


@app.command()
def export(run: Path = typer.Option(..., "--run"), output: Path = typer.Option(..., "--output")):
    """Copy a run's traceable artifacts into a paper_ready/ export."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "metrics").mkdir(parents=True, exist_ok=True)
    (output / "configs").mkdir(parents=True, exist_ok=True)
    (output / "manifests").mkdir(parents=True, exist_ok=True)

    fig_src = run / "reports" / "figures"
    if fig_src.exists():
        for f in fig_src.glob("*"):
            shutil.copy2(f, output / "figures" / f.name)
    eval_src = run / "evaluation"
    if eval_src.exists():
        for f in eval_src.glob("*.json"):
            shutil.copy2(f, output / "metrics" / f.name)
    for name in ("config.resolved.yaml",):
        p = run / name
        if p.exists():
            shutil.copy2(p, output / "configs" / name)
    for name in ("manifest.json", "dataset.json", "model.json", "dataset_profile.json"):
        p = run / name
        if p.exists():
            shutil.copy2(p, output / "manifests" / name)

    summary_src = run / "summary.json"
    experiment_id = json.loads(summary_src.read_text())["experiment_id"] if summary_src.exists() else "unknown"
    (output / "experiment_summary.md").write_text(
        f"# Paper-ready export\n\n- source run: `{run}`\n- experiment_id: `{experiment_id}`\n\n"
        "Every file under figures/, tables/, metrics/, configs/, manifests/ traces back to the "
        f"run directory above. Regenerate with `mtrain export --run {run} --output {output}`.\n"
    )
    console.print(f"exported to {output}")


@app.command("build-intervention")
def build_intervention_cmd(
    dataset: str = typer.Option(..., "--dataset"),
    version: str = typer.Option(..., "--version"),
    kind: str = typer.Option(..., "--kind"),
    variable: Optional[str] = typer.Option(None, "--variable"),
    variable_value: Optional[str] = typer.Option(None, "--variable-value"),
    oversample_factor: float = typer.Option(3.0, "--oversample-factor"),
    target_config: Path = typer.Option(..., "--target-config"),
    reason: str = typer.Option("", "--reason"),
    dry_run: bool = typer.Option(True, "--dry-run/--apply"),
):
    """Resolve an imbalanced/rebalanced intervention against a real dataset
    profile and write the resolved `intervention:` block into TARGET_CONFIG.

    Deliberately does not build the dataset version itself -- `mtrain train`
    does that as part of a normal run, with real lineage. This command only
    resolves *which* episodes/weights the intervention names, from the
    dataset's actual per-episode metadata, and never presumes them.
    """
    from rde.catalog import Catalog
    from rde.config import get_settings

    settings = get_settings()
    catalog = Catalog(settings.paths.catalog_db)
    rows = catalog.get_episodes(dataset, version)
    if not rows:
        console.print(f"[red]no episodes in {dataset}@{version}[/red]")
        raise typer.Exit(1)
    all_ids = [r["episode_id"] for r in rows]
    task_of = {r["episode_id"]: r["task_id"] for r in rows}
    from model_training.data.interventions import compute_length_bucket

    length_bucket_of = compute_length_bucket(rows)

    value: object = variable_value
    if variable == "task_id" and variable_value is not None:
        value = int(variable_value)

    def _target_episodes(var: Optional[str], val: object) -> list[int]:
        if var == "task_id":
            return [e for e in all_ids if task_of.get(e) == val]
        if var == "episode_length_bucket":
            return [e for e in all_ids if length_bucket_of.get(e) == val]
        return []

    if kind == "imbalanced":
        from model_training.data.interventions import resolve_episode_subset
        from model_training.config.schema import DatasetIntervention

        intervention = DatasetIntervention(kind="imbalanced", variable=variable, variable_value=value, reason=reason)
        episode_ids, resolved_reason = resolve_episode_subset(intervention, all_ids, task_of, length_bucket_of)
        block = {"kind": "custom", "episode_ids": episode_ids, "variable": variable,
                 "variable_value": value, "reason": resolved_reason}
    elif kind == "rebalanced":
        target = _target_episodes(variable, value)
        if not target:
            console.print(
                "[red]resolved zero episodes -- only --variable task_id / episode_length_bucket are "
                "auto-resolvable; pass explicit weights by hand otherwise[/red]"
            )
            raise typer.Exit(1)
        weights = {str(e): oversample_factor for e in target}
        block = {"kind": "rebalanced", "episode_weights": weights, "variable": variable,
                 "variable_value": value, "oversample_factor": oversample_factor,
                 "reason": reason or f"oversample {len(target)} episode(s) with {variable}={value}"}
    else:
        console.print(f"[red]build-intervention only resolves 'imbalanced'/'rebalanced'; got {kind!r}[/red]")
        raise typer.Exit(1)

    _print_json(block)
    if dry_run:
        console.print("[yellow]dry run -- pass --apply to write this into the target config[/yellow]")
        return

    data = yaml.safe_load(target_config.read_text()) if target_config.exists() else {}
    data["intervention"] = block
    target_config.write_text(yaml.safe_dump(data, sort_keys=False))
    console.print(f"wrote resolved intervention into {target_config}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
