# VLA Finetune

`vla-finetune` (Python package `model_training`, command-line interface `mtrain`) is the experiment, training, evaluation, deployment, and feedback layer for vision-language-action (VLA) policies. It is built on [`episodic-data-engine`](https://github.com/kushagrathisside/Episodic-Data-Engine-For-Cross-Embodiment-Data-Pipelines), which owns every dataset this package touches. Together the two repositories implement a data-centric development loop: the engine produces immutable, versioned datasets; this package fine-tunes and evaluates a policy against one of them; and the resulting failure analysis is returned as a structured recommendation for the next dataset version.

The package does not re-implement anything the data engine already provides. Ingestion, storage, quality control, versioning, lineage, and dataset construction remain the responsibility of `episodic-data-engine`, and are consumed through its public interface (`rde.engine.DataEngine`) and its model-facing modules (`rde.model_adapters`, `rde.evaluation`, `rde.feedback`).

<p align="center">
  <img src="docs/diagrams/01-system-architecture.svg" alt="System architecture: seven ExperimentRunner stages, the data-engine services each one calls, the artifacts each one writes, and the feedback loop to the next intervention" width="860">
</p>

**Figure 1. System architecture.** `ExperimentRunner.run` executes seven stages in order. Stages 2 to 6 call a data-engine service (left), and every stage writes named artifacts into the run directory (blue text). The feedback artifact's recommended action informs the next data intervention, which produces a new, lineage-tracked dataset version. The dashed arrow marks a decision taken by a person or a later run; it is not automatic.

## 1. Relationship to the Data Engine

Table 1 states the division of responsibility between the two repositories. The dependency runs in one direction only: `vla-finetune` imports `rde`, and `rde` imports nothing from `vla-finetune`.

**Table 1.** Division of responsibility.

| Responsibility | Owner |
|---|---|
| Ingestion, storage, catalog, quality control, versioning, lineage | `episodic-data-engine` |
| Dataset construction (splits, training contract) | `episodic-data-engine` |
| Model adapters, baseline policies, SmolVLA fine-tuning and inference | `episodic-data-engine` (`rde.model_adapters`) |
| Offline evaluation scoring | `episodic-data-engine` (`rde.evaluation`), extended with an `episode_ids` selector |
| Failure detection and episode nomination | `episodic-data-engine` (`rde.feedback`), reused unchanged |
| Experiment configuration and lifecycle orchestration | `vla-finetune` |
| Fine-tuning orchestration (checkpoint cadence, validation, model selection) | `vla-finetune` |
| Dataset profiling, split-integrity verification, data interventions | `vla-finetune` |
| Stratified evaluation and failure-artifact packaging | `vla-finetune` |
| Model artifact, local deployment runtime, reports | `vla-finetune` |

Three additive changes were made to the data engine to support this package: `SmolVLAPolicy.fit()` with checkpoint save, load, and verification; the `episode_ids` parameter on `Evaluator.evaluate()`; and the `EpisodeSubsetFilter` transform. They are documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §4.

## 2. Measured Results

Table 2 reports the three completed SmolVLA fine-tuning runs. All three start from the public `lerobot/smolvla_base` checkpoint and were trained on one NVIDIA GeForce RTX 4060 Laptop GPU (8 GB). Settings were batch size 4 and learning rate 1×10⁻⁴ with 50 warm-up steps. Each run used a single seed. The extended run resumes from the first run's step-1,000 checkpoint and continues for 1,000 more steps at a learning rate of 3×10⁻⁵. Test metrics are computed on each dataset's held-out test split, assigned by episode, so no test episode contributes any frame to training. The two SO-100 runs share an identical test split of 1,908 frames from 5 episodes.

**Table 2.** Offline test-split results. MSE is computed in normalized action units. The ratio is MSE divided by the MSE of the mean-action baseline on the same split.

| Run | Dataset | Steps | Test MSE | Ratio to mean-action | Nearest-neighbour MSE | Frames within tolerance¹ | Training time | Peak GPU memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_full` | SO-100 pick-place | 1,000 | 0.0290 | 3.2% | 0.0239 | 49.8% | 324 s | 2.2 GB |
| `baseline_full_extended` | SO-100 pick-place | 1,000 + 1,000 | 0.0239 | 2.6% | 0.0239 | 56.9% | 385 s | 2.2 GB |
| `aloha_coffee_baseline` | ALOHA coffee (bimanual, 14-DoF) | 1,000 | 0.2157 | 21.4% | 0.1062 | 0.05% | 576 s | 3.0 GB |

¹ This is the fraction of test frames whose mean absolute action error is below 0.1. It is a proxy defined in [`docs/METRICS.md`](docs/METRICS.md), not a measurement of task success.

The results support four statements, and no stronger ones.

1. Fine-tuning learns the task distribution. On SO-100, test MSE falls to 2.6–3.2% of the mean-action baseline.
2. Further training helped on SO-100. The extended run reduced test MSE by 18% (0.0290 → 0.0239) on the identical test split.
3. At this training budget, SmolVLA matches frame-level nearest-neighbour retrieval on SO-100 but does not beat it. On ALOHA coffee, retrieval is better by a factor of two. Retrieval is a strong baseline when test episodes come from the same setup as the training episodes. Beating it is the relevant target for further training, and these runs have not reached it.
4. These are open-loop, offline, single-seed measurements. No closed-loop rollout or physical-robot success rate is claimed (§7).

## 3. Installation

The data engine is consumed as an editable sibling checkout, not a published package, so both repositories must be cloned into the same parent directory. The second argument to each `git clone` sets the directory name that `pyproject.toml` (`../episodic-data-engine`) and the experiment runner expect:

```bash
git clone https://github.com/kushagrathisside/Episodic-Data-Engine-For-Cross-Embodiment-Data-Pipelines episodic-data-engine
git clone https://github.com/kushagrathisside/VLAFinetuningExperiments vla-finetune
cd vla-finetune
```

Two Python environments are used, both created in the shared parent directory. This mirrors the data engine's split between a lightweight base install and an optional dependency group for the VLA model (Table 3).

**Table 3.** Installation targets.

| Environment | Purpose | Commands |
|---|---|---|
| `../.venv` (light) | Configuration, profiling, interventions, tracking, reports, the CLI, and the unit and integration tests. No `torch`. | `uv venv --python 3.11 ../.venv`<br>`uv pip install --python ../.venv/bin/python -e ../episodic-data-engine -e ".[dev]"` |
| `../.venv-vla` (full) | Adds real SmolVLA fine-tuning and inference through the engine's `vla` extra (`torch`, `lerobot`). | `uv venv --python 3.11 ../.venv-vla`<br>`uv pip install --python ../.venv-vla/bin/python -e "../episodic-data-engine[vla]" -e ".[dev]"` |

Activate the environment that matches the task (`source ../.venv/bin/activate`, or `../.venv-vla/bin/activate` for training) before running `mtrain`. Datasets must first be ingested and versioned by the data engine (see its Quick Start). For example, `rde run --config configs/so100_pickplace.json` produces the `so100_pickplace@clean/v1` version that the shipped experiment configurations reference.

## 4. Command-Line Interface

The package installs the `mtrain` console script (Table 4). Each subcommand has an equivalent direct-invocation wrapper under `scripts/` for environments in which the console script is not on the path.

**Table 4.** `mtrain` subcommands.

| Command | Purpose |
|---|---|
| `inspect` | Print a dataset overview: episode, frame, and task counts, and storage footprint. |
| `profile` | Build and optionally persist a dataset profile, including coverage statistics. |
| `train` | Execute one experiment end to end (Figure 1, stages 1–7). |
| `validate` | Report the selected checkpoint's validation metrics for a completed run. |
| `evaluate` | Re-score a completed run's selected checkpoint against a specified split. |
| `test` | Equivalent to `evaluate --split test`. |
| `rollout` | Report closed-loop rollout status. This always reports "unavailable", because no simulator or robot is connected. |
| `compare` | Produce a comparison table across completed runs. |
| `analyze-failures` | Print the structured feedback artifact for a completed run. |
| `export` | Copy a run's traceable artifacts into a paper-ready export directory. |
| `build-intervention` | Resolve an imbalanced or rebalanced intervention from an actual dataset profile. |

Representative invocations:

```bash
mtrain profile --dataset so100_pickplace --version clean/v1
mtrain train --config configs/experiments/baseline.yaml
mtrain evaluate --run runs/<date>/baseline_full/seed_0 --split test
mtrain compare --runs runs/<date>/baseline_full/seed_0 --runs runs/<date>/reduced_data/seed_0
mtrain analyze-failures --run runs/<date>/baseline_full/seed_0
```

## 5. Run Directory

Every invocation of `mtrain train` creates `runs/<date>/<experiment_name>/seed_<seed>/` and refuses to overwrite a non-empty directory. Table 5 lists its contents. The `runs/` directory is excluded from version control.

**Table 5.** Run directory contents.

| Category | Files |
|---|---|
| Configuration and identity | `config.resolved.yaml`, `dataset.json`, `model.json`, `environment.json`, `git.json` |
| Data checks | `split_integrity.json`, `dataset_profile.json` |
| Training | `checkpoints/`, `metrics/train.jsonl`, `metrics/validation.jsonl`, `training_outcome.json`, and `crash.json` if the training loop fails |
| Evaluation | `evaluation/baselines.json`, `evaluation/test_overall.json`, `evaluation/stratified.json`, `metrics/test.jsonl`, `predictions/test_outcomes.jsonl` |
| Failure analysis and model | `failures/feedback_artifact.json`, `model_artifact.json` |
| Reports | `reports/experiment_report.md`, `reports/model_card.md`, `reports/figures/*.png`, `summary.json` |

Each run also appends rows to a cross-run index (`experiments/*.jsonl`, also excluded from version control). Every row is a single JSON object that can be queried with `jq`, `pandas`, or DuckDB.

## 6. Tests

```bash
../.venv/bin/python -m pytest tests -q          # 52 unit + integration tests; the smoke test skips without torch
../.venv-vla/bin/python -m pytest tests/smoke -q   # real SmolVLA fine-tuning steps; needs the vla extra
```

The unit and integration tests run against a synthetic dataset in the real LeRobot v3.0 layout, generated by the data engine's `rde.testing` module, with no network access. The smoke test runs a few real SmolVLA optimization steps on a tiny synthetic dataset. It is skipped, with a stated reason, when `torch` or `lerobot` is not installed.

## 7. Scope and Limitations

1. **Offline evaluation only.** All reported metrics are open-loop action-prediction errors on held-out demonstrations. `mtrain rollout` reports closed-loop evaluation as unavailable rather than estimating it.
2. **Single-seed, small-budget runs.** The results in Table 2 come from one seed and at most 2,000 optimization steps on a laptop GPU. They establish that the pipeline works end to end and give reference points; they are not a hyperparameter study.
3. **Single-node, local deployment.** `DeployedModel` is an in-process runtime with health checks and per-stage latency measurement. No network service, queue, or orchestration layer is implemented.
4. **Tracking-layer constraints.** The JSONL index has no file locking and is safe only for sequential runs. `ModelRegistry` is implemented and tested but is not called by `ExperimentRunner`. Only a training-loop failure writes a `crash.json` marker. All three are documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §8.

## 8. Further Documentation

**Table 6.** Documentation index.

| Document | Content |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Pipeline stages, the system boundary, the modifications made to the data engine, module organization, and known limitations. |
| [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) | Every CLI command, configuration field, and public Python interface, with common workflows. |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | The shipped experiment configurations, the choice of imbalance variable, evaluation slices, and seed policy. |
| [`docs/METRICS.md`](docs/METRICS.md) | Formal definitions of every computed metric, grouped by grain. |
| [`docs/FAILURE_ANALYSIS.md`](docs/FAILURE_ANALYSIS.md) | Failure extraction, ranking, stratification, and the evidentiary standard for candidate causes. |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | The model artifact format, loading procedure, health checks, and latency measurement. |
| [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) | Findings from the initial inspection of the data engine and the resulting implementation plan. |

## License

MIT. See [`LICENSE`](LICENSE).
