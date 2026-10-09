# Implementation Plan: Data-Centric VLA Training, Evaluation, Deployment, and Feedback

**Status.** Phase 1 (repository inspection and planning) is complete. This
document was produced before any training-layer code was written, in
accordance with the project requirement that the existing system be inspected
before extension.

## 1. Introduction

This document records the findings of an inspection of the existing
`episodic-data-engine` repository, states the architectural boundary this
project observes, and sets out the resulting implementation plan. It is
intended as a durable reference for the design decisions underlying the
`model_training` package, not a status log.

## 2. Findings: The Existing Data Engine

`episodic-data-engine` (Python package `rde`) is a mature, tested, and
documented data engine comprising 47 modules and 14,979 lines of code, with
a test suite of 188 tests (187 passing, 1 skipped) in the light environment
`../.venv`, which carries no heavy machine-learning dependency. Its
documentation series (`episodic-data-engine/docs/00`–`17`) was verified
against the source rather than assumed to be current.

The layers concerned with ingestion, storage, cataloging, validation,
transformation, and versioning are model-agnostic by construction: no module
under `rde/{ingestion,storage,catalog,validation,qc,synchronization,
transformations,versioning}` imports any model-specific code. This invariant
is preserved by the present work and was not modified.

More significantly, the engine already provides a substantial portion of the
model-facing functionality this project was asked to build. Table 1
summarizes the state of each component at the outset of this project.

**Table 1.** State of the model-facing layer prior to this project.

| Component | Location | Status |
|---|---|---|
| `ModelInput` / `Prediction` (mandatory confidence field) | `rde/model_adapters/base.py` | Complete |
| `StateAdapter`, `VisionStateAdapter`, `VisionLanguageAdapter` | `rde/model_adapters/adapters.py` | Complete; contract-driven, no hard-coded modality keys |
| `MeanActionPolicy`, `NearestNeighborPolicy` baselines | `rde/model_adapters/models.py` | Complete and tested; the nearest-neighbor baseline improves on the mean-action baseline by a factor of approximately thirty-eight |
| `SmolVLAPolicy` | `rde/model_adapters/models.py` | Inference only. `load()` and `predict()` were implemented; `fit()` was the inherited no-op, and no checkpointing existed. This was the principal gap this project was asked to close. |
| `MODEL_REGISTRY` / `build_model()` | `rde/model_adapters/models.py` | A name-to-class mapping; not a deployable-model ledger |
| Offline evaluation (mean squared error, mean absolute error, per-dimension error, success-rate proxy, baseline ratio) | `rde/evaluation/evaluator.py` | Complete at the aggregate and per-dimension grain; no stratification by task, variation, or episode-length bucket |
| Failure detection, episode nomination, quarantine, and oversampled retraining selection | `rde/feedback/flywheel.py` | Complete; this is the mechanism the specification's feedback-artifact requirement builds upon |
| Command-line interface (`rde model list/evaluate/feedback`) | `rde/cli.py` | Complete |
| Episode-level train/validation/test splitting, with quantile and independent modes and per-episode sampling weight | `rde/dataset_builder/builder.py` | Complete; this is the mechanism used for the data-intervention experiments |
| Lineage (transform runs, episode lineage, code fingerprinting) | `rde/versioning`, `rde/catalog` | Complete |
| Data-health contracts, drift detection, and observability | `rde/observability/*` | Complete; outside the scope of this project except as a source of distributional statistics |

The practical consequence of these findings is that this project is not the
construction of a training system alongside an independent data engine. It is
the addition of the missing components on top of a model-facing layer that
was already substantially built and tested: real fine-tuning, an experiment
abstraction, checkpointing, model selection, stratified and provenance-aware
failure analysis, a model registry, a deployment runtime, and reporting.
Duplicating any row of Table 1 would have reproduced the "second pipeline"
outcome the project specification explicitly prohibits.

### 2.1 Dataset Catalog State at Inspection Time

The catalog contained the following versions at the time of inspection.

**Table 2.** Dataset versions present in the catalog at inspection time.

| Dataset | Version | Episodes | Frames | Quality Status |
|---|---|---|---|---|
| `so100_pickplace` | `raw/v1` | 50 | 19,631 | warning |
| `so100_pickplace` | `clean/v1` | 48 | 18,668 | warning |
| `so100_pickplace` | `training/v1` | 48 | 18,668 | passed |
| `so100_pickplace` | `training/v2` | 48 | 18,668 | passed |
| `aloha_coffee` | `raw/v1` through `training/v1` | — | — | — |

`training/v1` is the version this project treats as the baseline for
`so100_pickplace`. `training/v2`'s provenance was independently verified via
`rde version lineage`: it is a second `build_training_dataset` run from the
same `clean/v1` (48 to 48 episodes, identical lineage roots to `v1`),
consistent with the flywheel demonstration (`make flywheel` or
`examples/06_feedback_flywheel.py`) rebuilding with different split or
weighting parameters. It was not treated as evidence of a data problem.

### 2.2 Execution Environment

The engine's own `.venv` carries no `torch` or `lerobot` dependency. A second
environment, `.venv-vla`, provides `torch` and `lerobot` under the engine's
optional `vla` extra. At inspection time this environment's `torch` build was
CPU-only (version 2.10.0+cpu), despite an NVIDIA RTX 4060 Laptop GPU (8 GB of
video memory) being visible to the host operating system via `nvidia-smi`
(driver version 592.82, CUDA 13.1). The host virtual machine provides
approximately 10 GB of RAM, consistent with the data engine's own design
target of an 8 GB machine (`scripts/capped` enforces a 6 GB ceiling by
default). A CUDA-enabled build of `torch` was subsequently installed to make
GPU-accelerated fine-tuning practical; this decision and its consequences are
recorded in Section 8.

The installed `lerobot` package (version 0.4.4) was inspected directly to
confirm the training entry points this project depends on:
`SmolVLAPolicy.forward(batch)` returns a loss dictionary, and
`SmolVLAPolicy.get_optim_params()` returns the model's trainable parameters.
These were verified against the installed source rather than assumed from
prior familiarity with the library.

## 3. System Boundary

The project observes the following boundary between the data engine and the
model-training layer.

```
Dataset Version -> DatasetView -> ModelInputAdapter -> Model -> Prediction -> Evaluation -> Feedback
```

`episodic-data-engine` owns every stage to the left of `DatasetView` and
depends on nothing to its right. `model_training` depends on
`episodic-data-engine` as an editable library dependency; the reverse
dependency does not exist and was not introduced.

### 3.1 Extensions Made to the Data Engine

Three additive changes were made to `episodic-data-engine`, each justified by
the specification's own instruction to extend an existing component rather
than duplicate it. Table 3 lists these changes.

**Table 3.** Additive changes made to `episodic-data-engine`.

| File | Change | Justification |
|---|---|---|
| `rde/model_adapters/models.py` | Added `SmolVLAPolicy.fit()` (real gradient-based fine-tuning), `save_checkpoint()`, `from_checkpoint()`, `verify_checkpoint()`, and `train_mode()`/`eval_mode()`. | The engine's own documentation states that adding a model is a matter of extending the corresponding class in this module. `SmolVLAPolicy` already existed as an inference-only adapter; extending it in place keeps `predict()` and `fit()` sharing one camera-mapping implementation rather than forking the inference path into a parallel class. |
| `rde/evaluation/evaluator.py` | Added an `episode_ids` parameter to `Evaluator.evaluate()`. | A generically useful capability, not specific to any model: it is the mechanism stratified evaluation (per-task, per-variation, or a fixed "hard cases" slice) is built on. |
| `rde/transformations/ops.py` | Added `EpisodeSubsetFilter`, a fourth episode filter alongside the existing `QualityFilter`, `LengthFilter`, and `TaskFilter`. | The mechanism by which reduced and imbalanced dataset variants become genuine, lineage-tracked `transform` runs, rather than an untraceable local copy. |

Each change is additive: no existing call site, test, or command-line
behavior was altered. The full `episodic-data-engine` test suite (188 tests)
passes unmodified after all three changes, and new tests were added
alongside each one.

### 3.2 Rationale for a Separate Package

`model_training` was implemented as a sibling package, checked out alongside
`episodic-data-engine` rather than nested inside it, for three reasons.

1. `episodic-data-engine` is its own git repository with an established,
   curated commit history and a numbered documentation series. Interleaving
   an unrelated concern into it would compromise that history.
2. The engine's base installation is deliberately lightweight: `torch` and
   `lerobot` are an optional extra, and dependencies such as `matplotlib` are
   absent entirely. A training, reporting, and deployment layer necessarily
   introduces heavier dependencies that do not belong in the engine's own
   dependency surface.
3. The separation mirrors the project specification's own boundary diagram,
   in which the model lifecycle is structurally a consumer of dataset
   versions rather than a peer within the engine that produces them.

## 4. Repository Structure

Table 4 lists the repository's top-level layout and the responsibility of
each module under `src/model_training/`.

**Table 4.** Repository layout.

| Path | Content |
|---|---|
| `pyproject.toml`, `README.md`, `LICENSE` | Package metadata, overview, and license. |
| `configs/base.yaml` | Defaults merged beneath every experiment configuration. |
| `configs/smolvla/so100_pickplace.yaml` | Model-specific settings for SmolVLA on the primary dataset. |
| `configs/experiments/*.yaml` | `baseline`, `baseline_extended`, `reduced`, `imbalanced`, `rebalanced`, and `aloha_coffee_baseline`. |
| `src/model_training/config/` | Experiment and training configuration schema. |
| `src/model_training/data/` | Dataset profiling, split-integrity verification, and interventions. |
| `src/model_training/training/` | Fine-tuning orchestration. |
| `src/model_training/evaluation/` | Stratified metrics, baselines, and error and temporal analysis. |
| `src/model_training/experiments/` | Experiment lifecycle orchestration and the data-fraction sweep. |
| `src/model_training/feedback/` | Structured feedback-artifact construction. |
| `src/model_training/deployment/` | Model artifact, registry, and local inference runtime. |
| `src/model_training/tracking/` | Run directories, reproducibility metadata, and the cross-run index. |
| `src/model_training/reports/` | Figures, experiment reports, model cards, and comparisons. |
| `src/model_training/cli.py` | The `mtrain` command-line interface. |
| `scripts/` | Direct-invocation wrappers around the CLI. |
| `tests/unit/`, `tests/integration/`, `tests/smoke/` | The three test tiers described in Section 9. |
| `docs/` | This document and its companions. |
| `runs/`, `experiments/` | Run directories and the cross-run index; created at run time and excluded from version control. |

## 5. Run Artifact Layout

Each experiment run produces a directory of the form
`runs/<date>/<experiment_name>/seed_<seed>/`, containing the following
categories of artifact.

**Table 5.** Run directory contents.

| Category | Files |
|---|---|
| Configuration and identity | `config.resolved.yaml`, `dataset.json`, `model.json`, `environment.json`, `git.json` |
| Metrics | `metrics/train.jsonl`, `metrics/validation.jsonl`, `metrics/test.jsonl`, `metrics/rollout.jsonl` |
| Model | `checkpoints/`, `model_artifact.json` |
| Evaluation | `evaluation/baselines.json`, `evaluation/test_overall.json`, `evaluation/stratified.json` |
| Provenance | `predictions/test_outcomes.jsonl`, `failures/feedback_artifact.json` |
| Reporting | `reports/experiment_report.md`, `reports/model_card.md`, `reports/figures/*.png` |

A `dataset.json` record for an intervention run states the parent version,
the selection rule applied, and the resulting `transform`/`build` run
identifiers from the data engine's own lineage tables, so that a run
directory is never a second source of truth about what the underlying data
was.

## 6. Experiment Lifecycle

The lifecycle implemented by `model_training.experiments.ExperimentRunner`
proceeds through the following stages.

1. Resolution of the experiment configuration and stamping of reproducibility
   metadata (git commit and dirty state for both repositories, Python and
   package versions, GPU identity where available).
2. Construction of the dataset intervention, via a real `transform` and
   `build_training_dataset` run when episode selection is required, or via
   per-episode sampling weight when the intervention is a rebalancing.
3. Independent re-verification of split integrity, prior to training.
4. Dataset profiling of the resulting `training/vN` version.
5. Fine-tuning, with periodic checkpointing and validation on the validation
   split only.
6. Selection of the best checkpoint by the configured validation metric.
7. Offline evaluation of the selected checkpoint on the test split, together
   with the baseline models, using the same evaluation call for both.
8. Stratified evaluation across the configured slices.
9. Failure detection and construction of the structured feedback artifact.
10. Generation of the experiment report, model card, and figures.

Closed-loop or rollout evaluation is reported as unavailable throughout,
since no simulator or physical robot is connected in this environment; the
specification's prohibition on converting offline prediction accuracy into a
claimed robot success rate is observed accordingly.

## 7. Selection of the Data-Intervention Variable

The project specification requires that the dimension along which the
imbalanced and rebalanced experiments are constructed be determined from an
actual profile of the dataset, rather than presumed in advance. Profiling
`so100_pickplace` confirms that the dataset carries a single task
instruction across all forty-eight episodes, which rules out task identity as
a usable variation axis. Trajectory length, by contrast, varies genuinely
(326 to 475 frames per episode, a mean duration of approximately thirteen
seconds with real spread), and is therefore the variable the imbalanced and
rebalanced experiment templates resolve to in practice. The mechanism
supporting this resolution, `episode_length_bucket`, is documented in
`docs/EXPERIMENTS.md`.

## 8. Decision Record: GPU Enablement

Two decisions with direct consequences for compute cost and elapsed time
were recorded explicitly rather than taken implicitly (Table 6).

**Table 6.** Compute decisions.

| Decision | Outcome | Rationale |
|---|---|---|
| Reinstall `torch` in `.venv-vla` with CUDA support | Adopted | The CPU-only build made fine-tuning impractical at the scale this project requires; a compatible GPU was already present but unused. |
| Initial compute budget | Build the full system, verify it with the smoke test, then run the full-data baseline experiment | Establishes a working, verified system before committing GPU time to the primary experiment. |

## 9. Testing Strategy

Three tiers of test are maintained.

1. **Unit tests** exercise configuration resolution, dataset-profile
   computation, the independent split-integrity check, stratified-metric
   arithmetic, checkpoint metadata, registry operations, and failure
   ranking, entirely within the light `.venv` environment.
2. **Integration tests** exercise the chain from `DataEngine` through
   `DatasetView`, the model adapter, the model, `Prediction`, the evaluator,
   stratified metrics, and the feedback artifact, using the baseline models
   so that the suite does not require the heavy dependency group.
3. **A smoke test** exercises the complete SmolVLA fine-tuning path—dataset
   resolution, batch construction, the forward and backward pass, checkpoint
   saving and reloading, evaluation, and report generation—against a tiny
   synthetic dataset. It requires the `vla` extra and is skipped with an
   explicit reason when that extra is unavailable, never replaced with a
   stub.

Explicit failure-mode tests cover invalid dataset identifiers and versions,
split leakage, unknown intervention variables, corrupted and incomplete
checkpoints, missing optional dependencies, and invalid deployment contracts,
each asserted to raise a specific and actionable error.

## 10. Known Limitations

1. Fine-tuning throughput depends on the CUDA installation described in
   Section 8; on a central-processing-unit-only build, fine-tuning is
   feasible only at small scale.
2. No simulator or physical robot is available in this environment. Every
   closed-loop or rollout metric is therefore structurally reported as
   unavailable; only offline evaluation is possible.
3. The host virtual machine provides approximately 10 GB of RAM, which bounds
   batch size and worker count independently of GPU availability.
4. `so100_pickplace` carries a single task instruction. Task-identity-based
   stratification and imbalance are supported for datasets that vary by
   task, but produce degenerate results on this dataset; trajectory length
   is used instead, as recorded in Section 7.

## 11. Next Steps

Implementation proceeds in the phased order set out in the project
specification, with the relevant test suite executed and its output
inspected after each phase. Results are reported as they are actually
obtained, not projected in advance.
