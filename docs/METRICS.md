# Metrics

## 1. Introduction

This document defines every metric computed by the system, grouped by the
grain at which it is measured: action level, episode level, dataset level,
closed-loop, and resource/inference level. For each metric, the table states
its definition, its intended interpretation, the split it is computed
against, and any known limitation.

## 2. Action-Level Metrics

Table 1 lists the metrics computed by `rde.evaluation.Evaluator.evaluate`,
reused without modification.

**Table 1.** Action-level metrics.

| Metric | Definition | Interpretation | Limitation |
|---|---|---|---|
| `mse` | Mean of the squared per-dimension error, across all frames and action dimensions. | Absolute error in squared, normalized-action units. | Not interpretable in isolation; must be read alongside the baseline ratio described in Section 7. |
| `mae` | Mean of the absolute per-dimension error. | Absolute error, in the same units as the action. | |
| `rmse` | Square root of `mse`. | Same units as the action; penalizes large individual errors more heavily than `mae`. | |
| `per_dimension_mae` | `mae` computed separately for each action dimension. | Identifies which control dimension the model is failing on; an aggregate figure can conceal a single systematically incorrect dimension. | |
| `normalized_mae` | Per-dimension `mae` divided by that dimension's own standard deviation over the evaluated frames, averaged across dimensions. | Comparable across dimensions whose natural ranges differ. | Dimensions with near-zero spread (below 10⁻³ of the median spread) are excluded, because dividing by them inflates the metric by orders of magnitude on small validation batches; `normalized_mae_dims` reports how many dimensions were kept. |
| `success_rate` | Fraction of frames with mean absolute error below a configurable tolerance (default 0.1). | An explicit proxy for task success, not a measurement of it. | Should never be reported without this qualification. |
| `mean_confidence`, `confidence_error_corr` | Mean of `Prediction.confidence`; Pearson correlation between confidence and error. | Meaningful for the nearest-neighbor baseline (a correlation of approximately −0.79 was measured on the test split). | For SmolVLA, confidence is an uncalibrated constant (recorded in `Prediction.metadata.confidence_note`), so this correlation carries no information for that model. |
| `latency_ms_per_frame` | Wall-clock time of each batched prediction from `Model.timed_predict`, divided by the batch size, averaged over batches. Excludes data loading and input adaptation. | Inference cost under the evaluation harness. | Not a substitute for the deployment latency breakdown described in Section 9. |

Baseline-relative metrics (`mse_ratio`, `mae_ratio`) are computed by
`rde.evaluation.Evaluator.compare` and by
`model_training.evaluation.offline.evaluate_baselines`, using the identical
evaluation call applied to the model under test. An absolute mean squared
error on normalized actions is close to meaningless in isolation; a model
that cannot outperform the training-set mean has, by definition, learned
nothing that this metric can detect.

## 3. Error-Distribution Metrics

`model_training.evaluation.offline.error_distribution` computes the median
and the 90th, 95th, and 99th percentiles, together with the minimum and
maximum, of per-frame mean absolute error. The metric is computed over
whichever split or slice was evaluated. Percentiles computed over a small
test set—this project's primary dataset has forty-eight episodes in
total—are statistically noisy; the frame count should always be reported
alongside the percentile values.

## 4. Episode-Level Metrics

`model_training.evaluation.offline.episode_metrics` aggregates the per-frame
outcomes already produced by `Evaluator.evaluate` into per-episode `mae`,
`mse`, `rmse`, and maximum error. Each is computed from the frame's scalar
error, the mean absolute error across action dimensions. The per-episode `mae`
therefore averages to the split-level `mae`, but the per-episode `mse` is the
mean of squared per-frame errors and is not comparable to the split-level
`mse`, which squares each dimension's error before averaging. The grain is one row per episode present in
the evaluated split or slice. These values are used both in report tables
and by `rde.feedback.FeedbackLoop`'s episode nomination.

## 5. Temporal Metrics

`model_training.evaluation.offline.temporal_error` bins mean error by
normalized episode progress—frame index divided by episode length minus
one—rather than by raw timestep, so that a 326-frame episode and a 475-frame
episode remain comparable at the same relative position. The function
re-reads, once per distinct episode already touched by the evaluation, the
small frame-index mapping via `DatasetView.get_episode_frames`; it does not
perform a second scan of the full dataset.

## 6. Dataset-Level and Coverage Metrics

`model_training.data.profile` computes episode, frame, and task counts, the
episode-length distribution, per-dimension action and state statistics (via
`rde.exploration.queries.DatasetExplorer`, reused without modification), the
modality list, split sizes, and a summary of quality-control findings.

Coverage metrics use the normalized Shannon entropy of the observed value
distribution, computed over ten equal-width bins per dimension: a value of
zero indicates that all observed mass falls in a single bin, and a value of
one indicates a uniform distribution across every non-empty bin. This
statistic measures how evenly the *observed* range of a dimension is used;
it is not, and cannot be, a claim about coverage of the underlying task
space.

Where a dimension requested by the project specification has no honest
computation available from this dataset's schema, the corresponding field is
reported explicitly as unavailable, together with the reason, rather than
approximated. Visual diversity is the one such field in the current
implementation: computing it would require an image embedding or similarity
model, which this profiler does not run.

## 7. Baseline Metrics

The mean-action baseline (which predicts the training-split mean regardless
of input, establishing the floor any model must exceed) and the
nearest-neighbor baseline (a retrieval policy whose confidence is derived
from a calibrated retrieval distance) are scored through the identical
`Evaluator.evaluate` call used for SmolVLA, in every experiment run, via the
`evaluation.baseline_models` configuration field.

## 8. Closed-Loop and Rollout Metrics

The project specification calls for success rate, completion time,
intervention count, and inference latency under closed-loop robot control.
None of these are computed in the present environment, because no simulator
or physical robot is connected. `mtrain rollout`, and the "Closed-loop /
rollout evaluation" section of every generated report, state this
explicitly as unavailable, together with the reason, rather than reporting a
number. Offline action-prediction accuracy is at no point presented as a
proxy for these metrics.

## 9. Resource and Inference Metrics

`model_training.deployment.DeployedModel.predict` reports the following
latency components, per invocation.

**Table 2.** Latency components reported by `DeployedModel.predict`.

| Component | Definition | Note |
|---|---|---|
| `model_load_ms` | Time to load the checkpoint. | Measured once, at `load()`. |
| `preprocessing_ms` | Image decode, resize, and tensor construction. | Reported as a single bucket; `rde.model_adapters.adapters.VisionStateAdapter` does not instrument these sub-stages individually, and this project does not fabricate a split it cannot measure. |
| `inference_ms` | The model forward pass. | Taken from `Model.timed_predict`, the same measurement used throughout evaluation. |
| `postprocessing_ms` | Everything else in the call. | |
| `total_ms` | The sum of the above. | |

`model_training.training.Trainer` additionally records `peak_rss_mb` (via
`psutil`), `peak_gpu_mb` (via `torch.cuda.max_memory_allocated`, reported as
unavailable when no compatible device is present), and `wall_clock_s` for
each training run.
