# Architecture

## 1. Introduction

This document describes the architecture of the `model_training` package: the
stages of the model lifecycle it implements, the boundary it maintains with
`episodic-data-engine`, the specific modifications made to the data engine in
support of this project, and the organization of the package's own modules.

## 2. Pipeline Overview

Figure 1 shows how `ExperimentRunner.run` orders the lifecycle, which
data-engine service each stage calls, and which artifacts each stage writes.
Table 1 then sets out the full lifecycle at a finer grain, from a versioned
dataset to a revised dataset version informed by model failure analysis,
together with the component responsible for each stage.

<p align="center">
  <img src="diagrams/01-system-architecture.svg" alt="System architecture: seven ExperimentRunner stages, the data-engine services each calls, the artifacts each writes, and the feedback loop" width="860">
</p>

**Figure 1.** System architecture. The dashed feedback arrow denotes a
decision taken by a person or by a later run, not an automatic step.

**Table 1.** Pipeline stages and owning components.

| Stage | Component | Notes |
|---|---|---|
| 1. Dataset version | `rde.catalog`, `rde.engine.DataEngine` | Immutable, catalog-resolved; the sole authoritative identity for the underlying data. |
| 2. Dataset profile | `model_training.data.profile.build_dataset_profile` | Built on `rde.exploration.queries.DatasetExplorer`; adds coverage and diversity statistics. |
| 3. Research experiment | `model_training.config.ExperimentConfig` | Dataset reference, intervention, splits, training configuration, evaluation configuration, and tags. |
| 4. Data intervention (conditional) | `model_training.data.interventions.build_intervention_dataset` | Invoked when the intervention is not the full dataset. Constructs a genuine `rde` transform and dataset-build run. |
| 5. Dataset view | `rde.engine.DataEngine.get_dataset` | Unmodified. |
| 6. Vision-language adapter | `rde.model_adapters.VisionLanguageAdapter` | Reused without modification; reads the training contract rather than hard-coding modality keys. |
| 7. SmolVLA | `rde.model_adapters.models.SmolVLAPolicy` | Extended in place: `fit()` now performs real gradient steps; `save_checkpoint()` and `from_checkpoint()` were added. |
| 8. Training | `model_training.training.Trainer` | Orchestrates checkpointing cadence, periodic validation, resource sampling, and metrics logging around `SmolVLAPolicy.fit()`, which owns the mechanics of a single optimization step. `policy.fit(...)` runs inside a `try`/`except`: a training-loop crash (OOM, a NaN loss, a signal) writes `crash.json` (error type/message, steps completed, checkpoints so far, wall-clock time) before re-raising, and releases GPU memory (`del policy; torch.cuda.empty_cache()`). A crash does not produce a final checkpoint — that guarantee holds only for a run that completes `policy.fit()` normally. |
| 9. Validation | The same training loop, scored via `rde.evaluation.Evaluator` restricted to the validation split | Never the test split. |
| 10. Model selection | `model_training.training.Trainer` | The checkpoint with the best validation metric, restricted to checkpoints that passed `SmolVLAPolicy.verify_checkpoint` immediately after saving — an unverified save is recorded with `"verified": false` and is never eligible as `best`, so a process killed mid-write cannot get preserved by the pruning logic that always protects the current best. |
| 11. Frozen model | `model_training.deployment.ModelArtifact` | The deployable identity of the selected checkpoint. `model_training.deployment.ModelRegistry` exists (Section 6) but `ExperimentRunner` does not write through it; the model is actually looked up via `ExperimentIndex.models` (`experiments/models.jsonl`) — see Section 6 and `docs/DEPLOYMENT.md`. |
| 12. Offline test / rollout | `rde.evaluation.Evaluator` (test split) and `model_training.evaluation.offline` | Rollout is reported as unavailable; no simulator or robot is connected in this environment. |
| 13. Metrics | `runs/.../metrics/*.jsonl`, `evaluation/*.json`, and the cross-run index | |
| 14. Failure analysis | `rde.feedback.FeedbackLoop`, wrapped by `model_training.feedback.build_feedback_artifact` | |
| 15. Data feedback | The feedback artifact | Ranked episodes, candidate explanations, and a recommended next action; never presumes the data is at fault. |
| 16. New dataset version | A subsequent `build_intervention_dataset` run, informed by the feedback artifact | Closes the loop. |

## 3. System Boundary

The project maintains the following interface between the data engine and
the model-training layer:

```
Dataset Version -> DatasetView -> ModelInputAdapter -> Model -> Prediction -> Evaluation -> Feedback
```

`episodic-data-engine` owns every stage to the left of `DatasetView` and
imports nothing to its right. `model_training` depends on
`episodic-data-engine` as an editable library dependency, declared in
`pyproject.toml` under `[tool.uv.sources]`; the reverse dependency does not
exist. This relationship was true before this project began and remains true
after it.

## 4. Modifications to the Data Engine

Table 2 lists the three additive changes made to `episodic-data-engine` in
support of this project. Each is justified individually below the table.

**Table 2.** Modifications made to `episodic-data-engine`.

| File | Change |
|---|---|
| `rde/model_adapters/models.py` | `SmolVLAPolicy.fit()` (real gradient-based fine-tuning), `save_checkpoint()`, `from_checkpoint()`, `verify_checkpoint()`, `train_mode()`, `eval_mode()` |
| `rde/evaluation/evaluator.py` | An `episode_ids` parameter on `Evaluator.evaluate()` |
| `rde/transformations/ops.py` | `EpisodeSubsetFilter`, a fourth episode filter |

The engine's own documentation states that adding a model to the system is a
matter of extending the corresponding class in `rde/model_adapters/models.py`.
`SmolVLAPolicy` already existed there as an inference-only adapter; extending
it in place, rather than introducing a parallel `TrainableSmolVLAPolicy`
class, keeps `predict()` and `fit()` sharing a single camera-mapping
implementation and so cannot diverge from one another.

The `episode_ids` parameter added to `Evaluator.evaluate()` is a generically
useful capability, not specific to any one model: it is the mechanism by
which stratified evaluation—by task, by variation, or against a fixed "hard
cases" slice—selects an explicit episode subset rather than a named split.

`EpisodeSubsetFilter` is the mechanism by which the reduced and imbalanced
dataset variants become genuine, lineage-tracked `rde` transform runs, rather
than an untraceable local copy of the data.

Every change is additive. No existing call site, test, or command-line
behavior in `episodic-data-engine` was altered, and the engine's full test
suite (188 tests: 187 passing, 1 skipped) passes unmodified after all three
changes. Each change is accompanied by new tests in
`episodic-data-engine/tests/`.

## 5. Rationale for a Separate Package

`model_training` is implemented as a sibling package rather than as a
subdirectory of `episodic-data-engine`, for three reasons.

1. `episodic-data-engine` is its own git repository with an established
   commit history and a numbered documentation series. Interleaving an
   unrelated concern into it would compromise that history.
2. The engine's base installation is deliberately lightweight: `torch` and
   `lerobot` are declared as an optional extra, and `matplotlib` is not a
   dependency of the engine at all. A training, reporting, and deployment
   layer necessarily introduces heavier dependencies that do not belong in
   the engine's dependency surface.
3. The separation mirrors the project specification's own boundary, in which
   the model lifecycle is structurally a consumer of dataset versions rather
   than a peer within the engine that produces them.

## 6. Module Organization

Table 3 summarizes the responsibility of each module under
`src/model_training/`.

**Table 3.** Module responsibilities.

| Module | Responsibility |
|---|---|
| `config/` | `ExperimentConfig`, `TrainingConfig`, `EvaluationConfig`, and `DatasetIntervention`; validated with `pydantic` and loaded from YAML. |
| `data/` | Dataset profiling, independent split-integrity verification, and intervention (reduced, imbalanced, rebalanced) construction. |
| `training/` | The `Trainer`, orchestrating checkpointing, validation cadence, and resource sampling around `SmolVLAPolicy.fit()`. |
| `evaluation/` | Stratified metrics, baseline scoring, error-distribution and temporal analysis, all built on `rde.evaluation.Evaluator`. |
| `experiments/` | `ExperimentRunner`, the full lifecycle described in Table 1, and the data-fraction sweep driver. |
| `feedback/` | Construction of the structured feedback artifact from `rde.feedback.FeedbackLoop`'s output. |
| `deployment/` | `ModelArtifact`, `DeployedModel` (the local inference runtime with health checks and latency measurement), and `ModelRegistry` (implemented, tested, but not wired into `ExperimentRunner`; see Section 8). |
| `tracking/` | `RunDir` and reproducibility metadata collection; the cross-run JSONL index (`JsonlIndex`, append-only, no file locking — see Section 8). |
| `reports/` | Figure generation, the experiment report and model card templates, and comparison tables. |
| `cli.py` | The `mtrain` command-line interface. |

## 7. Execution Environments

Table 4 describes the two Python environments the package is designed
against.

**Table 4.** Execution environments.

| Environment | Contents | Used for |
|---|---|---|
| `../.venv` | `episodic-data-engine` and the light layer of `model_training` (configuration, data, tracking, reports, deployment health checks, and the command-line interface). No `torch`. | The unit and integration test suites; `mtrain profile`, `mtrain inspect`, and `mtrain build-intervention`. |
| `../.venv-vla` | Adds `episodic-data-engine[vla]` (`torch` and `lerobot`). | `mtrain train` (real fine-tuning), evaluation of a SmolVLA checkpoint, and the smoke test's fine-tuning step. |

Neither environment installs a dependency the other does not need; this
mirrors the data engine's own separation exactly.

## 8. Known Limitations in the Tracking and Deployment Layer

1. **`ModelRegistry` (`model_training/deployment/registry.py`) is not the
   actual lookup path.** `ExperimentRunner.run` never calls
   `ModelRegistry.register`; it writes model records directly via
   `ExperimentIndex.record_model`, i.e. `experiments/models.jsonl`.
   `ModelRegistry` uses the same `JsonlIndex` row format, so when it is
   pointed at `experiments/models.jsonl` it sees the runner's records (its
   `get` returns the most recent row for a re-registered `model_id`). It is
   covered by unit tests but is not called by the runner.
2. **`JsonlIndex` has no file locking.** Two `mtrain train` invocations
   writing to the same index file concurrently can interleave partial
   lines. Safe only for sequential runs.
3. **GPU memory is released after training, but crash-safety is scoped to
   the training loop only.** `Trainer.train` frees the policy
   (`del policy; torch.cuda.empty_cache()`) on both the success and the
   exception path, so `ExperimentRunner.run`'s subsequent construction of a
   second `SmolVLAPolicy` for evaluation does not contend with the first for
   GPU memory. There is no equivalent run-status marker at the
   `ExperimentRunner.run()` level: only a training-loop failure produces
   `crash.json`. A crash during data intervention, evaluation, or report
   generation still leaves an unmarked, partially populated run directory.
