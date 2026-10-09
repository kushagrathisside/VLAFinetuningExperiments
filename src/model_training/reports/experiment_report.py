"""``experiment_report.md`` generation -- assembled entirely from
saved run artifacts, never from mutable live state, so re-running this
function against a finished run's directory always reproduces the same
report."""

from __future__ import annotations

from typing import Any


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


def _metrics_table(metrics: dict[str, Any]) -> str:
    rows = [(k, v) for k, v in metrics.items() if not isinstance(v, (list, dict))]
    if not rows:
        return "_no scalar metrics recorded_\n"
    lines = ["| metric | value |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in rows]
    return "\n".join(lines) + "\n"


def render_experiment_report(ctx: dict[str, Any]) -> str:
    """``ctx`` carries everything the report cites; see
    ``model_training.experiments.runner.ExperimentRunner`` for exactly what it
    assembles. Fields that were not run/available render as stated text
    ("not run", "unavailable") rather than being omitted silently."""
    cfg = ctx["config"]
    lines: list[str] = []
    lines.append(f"# Experiment report: {cfg['experiment_name']}\n")
    lines.append(f"- **experiment_id**: `{cfg['experiment_id']}`")
    lines.append(f"- **run_dir**: `{ctx['run_dir']}`")
    lines.append(f"- **generated_at**: {ctx['generated_at']}\n")

    lines.append("## Objective and hypothesis\n")
    lines.append(f"**Objective:** {cfg.get('objective') or '_none stated_'}\n")
    lines.append(f"**Hypothesis:** {cfg.get('hypothesis') or '_none stated_'}\n")

    lines.append("## Dataset\n")
    ds = ctx["dataset"]
    lines.append(f"- input version: `{ds['dataset_id']}@{ds['input_version']}`")
    lines.append(f"- clean version used: `{ds['clean_version']}`")
    lines.append(f"- training version (this run's authoritative identity): `{ds['training_version']}`")
    lines.append(f"- intervention: `{cfg['intervention']['kind']}` -- {ds.get('selection_reason', 'n/a')}")
    lines.append(f"- episodes used: {len(ds.get('episode_ids', []))}  "
                 f"(excluded: {len(ds.get('excluded_episode_ids', []))})\n")

    if ctx.get("profile"):
        p = ctx["profile"]
        lines.append("### Dataset statistics (from profile)\n")
        lines.append(f"- episodes: {p['episode_count']}, frames: {p['frame_count']}, tasks: {p['task_count']}")
        lines.append(f"- splits: {p.get('splits')}")
        cov = p.get("coverage", {})
        lines.append(f"- action-space coverage (mean normalized entropy): "
                     f"{_fmt(cov.get('action_space_coverage', {}).get('mean_normalized_entropy'))}")
        lines.append(f"- instruction diversity (normalized entropy): "
                     f"{_fmt(cov.get('instruction_diversity', {}).get('diversity_normalized_entropy'))}")
        lines.append(f"- visual diversity: {cov.get('visual_diversity', {}).get('status', 'unavailable')}\n")

    lines.append("## Model\n")
    model = ctx.get("model", {})
    lines.append(f"- architecture: {model.get('architecture', cfg['training']['model'])}")
    lines.append(f"- pretrained checkpoint: {cfg['training']['pretrained_checkpoint']}")
    lines.append(f"- selected checkpoint: `{model.get('selected_checkpoint', 'n/a')}`")
    lines.append(f"- selection metric ({cfg['evaluation']['selection_metric']}, "
                 f"{cfg['evaluation']['selection_mode']}): {_fmt(model.get('selection_value'))}\n")

    lines.append("## Training configuration\n")
    tc = cfg["training"]
    lines.append(f"- steps: {tc['steps']}, batch_size: {tc['batch_size']}, lr: {tc['learning_rate']}, "
                 f"grad_accum: {tc['grad_accum_steps']}, seed: {tc['seed']}")
    outcome = ctx.get("training_outcome")
    if outcome:
        lines.append(f"- steps completed: {outcome['steps_completed']}, "
                     f"final train loss: {_fmt(outcome['final_train_loss'])}")
        lines.append(f"- wall clock: {_fmt(outcome['wall_clock_s'], 3)} s, "
                     f"peak RSS: {_fmt(outcome.get('peak_rss_mb'), 3)} MB, "
                     f"peak GPU: {_fmt(outcome.get('peak_gpu_mb'), 3)} MB\n")
    else:
        lines.append("- training: **not run** in this report context\n")

    lines.append("## Compute environment\n")
    env = ctx.get("environment", {})
    lines.append(f"- torch: {env.get('torch') or 'not installed'}, lerobot: {env.get('lerobot') or 'not installed'}")
    gpu = env.get("gpu")
    lines.append(f"- GPU: {gpu['name'] if gpu else 'unavailable'}")
    lines.append(f"- git (episodic-data-engine): {env.get('git', {}).get('episodic_data_engine', {}).get('commit', 'n/a')}")
    lines.append(f"- git (model_training): {env.get('git', {}).get('model_training', {}).get('commit', 'n/a')}\n")

    lines.append("## Test metrics (fixed test split)\n")
    test = ctx.get("test_metrics")
    if test:
        lines.append(_metrics_table(test["overall"]))
    else:
        lines.append("_not run_\n")

    lines.append("## Baseline comparison\n")
    baselines = ctx.get("baseline_metrics")
    if baselines:
        lines.append("| model | mse | mae | vs mean_action (mse ratio) |")
        lines.append("|---|---|---|---|")
        mean_mse = baselines.get("mean_action", {}).get("mse")
        for name, m in baselines.items():
            ratio = (m["mse"] / mean_mse) if mean_mse else None
            lines.append(f"| {name} | {_fmt(m.get('mse'))} | {_fmt(m.get('mae'))} | {_fmt(ratio)} |")
        if ctx.get("model_test_mse") is not None and mean_mse:
            lines.append(f"| **{cfg['training']['model']} (this run)** | {_fmt(ctx['model_test_mse'])} | "
                         f"n/a | {_fmt(ctx['model_test_mse'] / mean_mse)} |")
        lines.append("")
    else:
        lines.append("_not run_\n")

    lines.append("## Stratified metrics\n")
    strat = ctx.get("stratified_metrics")
    if strat:
        lines.append("| slice | frames | mse | mae |")
        lines.append("|---|---|---|---|")
        for name, m in strat.items():
            lines.append(f"| {name} | {m.get('frames')} | {_fmt(m.get('mse'))} | {_fmt(m.get('mae'))} |")
        lines.append("")
    else:
        lines.append("_not run_\n")

    lines.append("## Failure analysis\n")
    fb = ctx.get("feedback_artifact")
    if fb:
        lines.append(f"- episodes flagged: {len(fb.get('ranked_episode_reviews', []))}")
        for note in fb.get("candidate_explanations", []):
            lines.append(f"  - {note}")
        lines.append(f"- recommended next action: {fb.get('recommended_next_action')}\n")
    else:
        lines.append("_not run_\n")

    lines.append("## Closed-loop / rollout evaluation\n")
    lines.append(f"{ctx.get('rollout_status', 'unavailable -- no simulator or robot connected in this environment')}\n")

    lines.append("## Limitations\n")
    for note in ctx.get("limitations", []) or ["_none recorded_"]:
        lines.append(f"- {note}")
    lines.append("")

    lines.append("## Conclusion\n")
    lines.append(ctx.get("conclusion") or "_not yet written -- fill in after reviewing the metrics above._\n")

    lines.append("## Data feedback recommendation\n")
    lines.append(fb.get("recommended_next_action") if fb else "_not run_")

    return "\n".join(lines) + "\n"
