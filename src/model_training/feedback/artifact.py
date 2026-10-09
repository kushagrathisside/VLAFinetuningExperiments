"""The structured feedback artifact.

Wraps ``rde.feedback.FeedbackLoop`` -- the engine's own failure-detection and
episode-nomination mechanism -- rather than reimplementing it, and adds what
that module deliberately leaves to a human: candidate explanations framed
against the rule that "a model failure does not automatically imply bad
data", and a concrete recommended next action a curator or this project's own
``build_intervention_dataset(kind="rebalanced")`` can act on directly.
"""

from __future__ import annotations

from typing import Any

from rde.catalog import Catalog
from rde.config import Settings
from rde.evaluation.evaluator import EvaluationResult
from rde.feedback import FeedbackLoop


def build_feedback_artifact(
    experiment_id: str,
    model_id: str,
    dataset_id: str,
    dataset_version: str,
    result: EvaluationResult,
    *,
    settings: Settings | None = None,
    catalog: Catalog | None = None,
    error_percentile: float = 90.0,
    min_failures_per_episode: int = 3,
) -> dict[str, Any]:
    """Detect failures via the real flywheel, then package the structured artifact."""
    loop = FeedbackLoop(settings=settings, catalog=catalog)
    report = loop.detect(
        result, error_percentile=error_percentile,
        min_failures_per_episode=min_failures_per_episode, persist=True,
    )
    selection = loop.build_retraining_selection(report)

    candidate_explanations: list[str] = []
    if report.reviews:
        concentrated = [r for r in report.reviews if r.failure_rate > 0.5]
        if concentrated:
            candidate_explanations.append(
                f"{len(concentrated)} episode(s) fail on more than half their evaluated frames -- "
                "concentrated, episode-local failure is evidence worth checking for a data/collection "
                "defect in those specific episodes (corruption, mislabeling, visual ambiguity), not just "
                "model weakness (a model failure does not automatically imply bad data)."
            )
        scattered = [r for r in report.reviews if r.failure_rate <= 0.5]
        if scattered:
            candidate_explanations.append(
                f"{len(scattered)} episode(s) fail on a minority of their evaluated frames -- more "
                "consistent with a model/coverage limitation (insufficient training exposure to that "
                "state region, or a genuinely hard sub-task) than an episode-specific data defect."
            )
    else:
        candidate_explanations.append(
            "no episode crossed the failure-concentration threshold at this percentile/min-failures "
            "setting; no evidence of either a data or model problem concentrated enough to flag."
        )

    recommended = (
        "no action indicated by this evaluation."
        if not report.reviews else
        "inspect the top-ranked episode(s) below for corruption/ambiguity before assuming a data fix "
        "will help; if the pattern holds and traces to a real shared attribute (task, length bucket), "
        "build a targeted correction with "
        "`mtrain build-intervention --kind rebalanced --variable <attribute> --variable-value <value>` "
        "and retrain, then compare against this run on the same fixed test slice."
    )

    return {
        "experiment_id": experiment_id,
        "model_id": model_id,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "affected_slice": {"split": result.split, "frames_evaluated": result.frames_evaluated},
        "failure_summary": report.summary(),
        "ranked_episode_reviews": [r.to_dict() for r in report.reviews],
        "candidate_explanations": candidate_explanations,
        "recommended_next_action": recommended,
        "retraining_selection": selection,
        "rde_artifacts": report.artifacts,
    }
