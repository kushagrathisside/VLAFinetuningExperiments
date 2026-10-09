"""``model_card.md`` generation. Claims only what evaluation
actually measured -- no capability beyond what ``test_metrics``/``rollout``
carry is stated."""

from __future__ import annotations

from typing import Any


def render_model_card(artifact: dict[str, Any], ctx: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Model card: {artifact['model_id']}\n")
    lines.append(f"- **architecture**: {artifact['architecture']}")
    lines.append(f"- **pretrained base**: {artifact['pretrained_model']}")
    lines.append(f"- **version**: {artifact['model_version']}")
    lines.append(f"- **training dataset**: `{artifact['dataset_id']}@{artifact['dataset_version']}`")
    lines.append(f"- **embodiment**: {artifact.get('embodiment') or 'unspecified'}")
    lines.append(f"- **created**: {artifact['created_at']}")
    lines.append(f"- **experiment**: `{artifact['experiment_id']}`\n")

    lines.append("## Input modalities\n")
    obs = artifact.get("observation_schema", {})
    images = [k for k, v in obs.items() if v.get("plane") == "external"]
    state = [k for k, v in obs.items() if v.get("plane") == "inline"]
    lines.append(f"- cameras: {', '.join(images) or 'none'}")
    lines.append(f"- state: {', '.join(state) or 'none'}")
    lines.append(f"- action dimension: {artifact.get('action_schema', {}).get('shape')}\n")

    lines.append("## Training configuration\n")
    tc = ctx.get("training_config", {})
    for k in ("steps", "batch_size", "learning_rate", "seed"):
        if k in tc:
            lines.append(f"- {k}: {tc[k]}")
    lines.append("")

    lines.append("## Evaluation performance\n")
    metrics = artifact.get("metrics") or {}
    if metrics:
        for k, v in metrics.items():
            if not isinstance(v, (list, dict)):
                lines.append(f"- {k}: {v}")
    else:
        lines.append("_no evaluation metrics recorded on this artifact_")
    lines.append("")

    lines.append("## Known failure modes\n")
    failures = ctx.get("failure_notes") or []
    if failures:
        for f in failures:
            lines.append(f"- {f}")
    else:
        lines.append("_no failure analysis attached to this artifact_")
    lines.append("")

    lines.append("## Deployment requirements\n")
    deps = artifact.get("dependencies", {})
    lines.append(f"- torch: {deps.get('torch') or 'required, version unrecorded'}")
    lines.append(f"- lerobot: {deps.get('lerobot') or 'required, version unrecorded'}")
    lines.append(f"- loader: `model_training.deployment.DeployedModel` (local, no server)\n")

    lines.append("## Limitations\n")
    lines.append("- confidence is an uncalibrated constant (see `Prediction.metadata.confidence_note`); "
                 "not usable as a real uncertainty signal.")
    lines.append("- no closed-loop/rollout evaluation exists for this artifact -- "
                 "reported metrics are offline (predicted vs. demonstrated action), not task success.")
    for note in ctx.get("limitations", []) or []:
        lines.append(f"- {note}")

    return "\n".join(lines) + "\n"
