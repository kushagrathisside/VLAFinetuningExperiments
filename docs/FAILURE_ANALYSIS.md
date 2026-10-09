# Failure Analysis

## 1. Introduction

This document describes the failure-extraction and ranking methodology
implemented by `model_training.feedback`, the dimensions along which
failures are stratified, the provenance recorded for each flagged
prediction, the evidentiary standard applied when proposing a candidate
cause, and the procedure by which a feedback artifact leads to a revised
dataset version.

## 2. Failure Extraction and Ranking

`model_training.feedback.build_feedback_artifact` wraps
`rde.feedback.FeedbackLoop.detect`, reused without modification, rather than
reimplementing failure detection. A frame is flagged as `HIGH_ERROR`,
`LOW_CONFIDENCE`, or `BOTH` against percentiles computed from the current
evaluation run, not a fixed absolute threshold; a fixed threshold would flag
either everything or nothing the moment the model or the normalization
scheme changes. The confidence criterion applies only when a model's
confidence varies across frames. SmolVLA reports a constant confidence, so its
failures are judged on error alone; the run's `thresholds` record this as
`confidence_used: false`. Before this rule existed, a constant confidence
placed every frame at its own percentile and flagged the entire test split.
`FeedbackLoop`'s internal nomination procedure aggregates
frame-level flags into episode-level reviews, ranked by a priority score
equal to the product of the failure rate and one plus the mean error, so
that an episode failing often and severely outranks one with a small number
of isolated error spikes.

## 3. Stratification Dimensions

Beyond `FeedbackLoop`'s own per-episode ranking, this project adds the
dimensions listed in Table 1.

**Table 1.** Additional stratification dimensions.

| Dimension | Source | Purpose |
|---|---|---|
| Per-action-dimension error | `Evaluator`'s `per_dimension_mae`, surfaced per episode via `episode_metrics` | Identifies which control dimension is responsible for a given episode's failure. |
| Temporal region | `model_training.evaluation.offline.temporal_error` | Distinguishes failure concentrated near episode starts (often an ambiguous initial state) from failure concentrated near episode ends (often accumulated drift). |
| Highest-error episodes and frames | `episode_metrics`, sorted by mean squared error, and the raw per-frame outcomes persisted to `predictions/test_outcomes.jsonl` | Supports direct inspection of the worst-performing cases. |

## 4. Provenance and Traceability

Every persisted outcome row carries the run identifier, experiment
identifier, dataset identifier and version, frame identifier, episode
identifier, error, confidence, and per-dimension error—sufficient to trace a
prediction back through its frame and episode to the dataset version and its
lineage, via `rde version trace <dataset_id> <training_version>`.

One limitation is recorded here explicitly: the raw predicted-action and
ground-truth-action vectors are not persisted for every evaluated frame,
only the derived error. Persisting the full vectors for every frame in every
evaluation would materially increase the size of each run directory, and
because predictions are deterministic given the checkpoint and the frame,
they are reproducible on demand rather than stored redundantly. Where the
raw vectors are required for a specific investigation, `mtrain evaluate` can
be re-run scoped to the episode or episodes in question.

## 5. Candidate Causes and Evidentiary Standards

The project specification requires that a model failure not be treated as
automatic evidence of a data defect. `build_feedback_artifact` frames its
candidate explanations accordingly, distinguishing two patterns.

1. **Concentrated failure**, in which an episode fails on more than half of
   its evaluated frames, is reported as evidence worth investigating for a
   data or collection defect specific to that episode—corruption,
   mislabeling, or visual ambiguity—without asserting that such a defect is
   present.
2. **Scattered failure**, in which an episode fails on a minority of its
   frames, is reported as more consistent with a model or coverage
   limitation than with an episode-specific defect.

Neither pattern is presented as a conclusion; both are handed to a human
reviewer, or to the hypothesis of a subsequent experiment, as evidence. No
part of this pipeline deletes or excludes an episode on the basis of model
error. `FeedbackLoop.quarantine_episodes` marks an episode for human review
in the catalog; it does not remove the episode from any dataset version, and
every subsequent `training/vN+1` continues to be built from the full set of
real episodes, with under-performing episodes upweighted—never dropped—via
the `episode_weights` mechanism.

## 6. From Feedback Artifact to Dataset Revision

The `recommended_next_action` field of the feedback artifact names a
concrete next step whenever the evidence supports one:

```bash
mtrain build-intervention --kind rebalanced \
  --variable <attribute> --variable-value <value> \
  --target-config configs/experiments/rebalanced.yaml --apply
mtrain train --config configs/experiments/rebalanced.yaml
mtrain compare --runs <previous run> --runs <new run>
```

This sequence closes the loop described in the project specification—from a
dataset version, through model training and evaluation, to a failure slice,
a revised dataset version, retraining, and re-evaluation—with each stage
represented by a real, persisted artifact rather than an implicit or
in-memory decision.
