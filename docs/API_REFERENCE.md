# API Reference

## 1. Purpose and Scope

This document is the single reference for every entry point in `model_training`
relevant to training-related work: the command-line interface, the
experiment-configuration schema, the Python API this package exposes, and the
`episodic-data-engine` (`rde`) APIs a training or evaluation script calls
directly. It exists for knowledge transfer — a team member who has not touched
this codebase should be able to run an experiment, read its output, write a
new one, or extend the system to a new model or dataset using this document
alone, without re-deriving signatures from source.

This document does not repeat the design rationale already written elsewhere.
Table 1 in Section 11 points to where each adjacent question is answered
instead. What follows here is deliberately just the interface: what exists,
what it takes, what it returns, and where its output lands.

**Table 1.** Scope note.

| This document covers | This document does not cover |
|---|---|
| `mtrain` CLI commands and flags | Why the four intervention kinds exist (see `EXPERIMENTS.md`) |
| `ExperimentConfig` and its sub-schemas, field by field | Metric definitions and their statistical meaning (see `METRICS.md`) |
| Every public function/class in `model_training`, by module | The deployment runtime's design rationale (see `DEPLOYMENT.md`) |
| `rde` APIs consumed directly by training/evaluation code | `rde`'s own ingestion, storage, QC and versioning internals (see `episodic-data-engine/docs/`) |
| Run-directory artifact layout | The failure-detection algorithm itself (see `FAILURE_ANALYSIS.md`) |

## 2. Environment Setup

The project splits across two Python virtual environments, mirroring
`episodic-data-engine`'s own split between its core and its optional `vla`
extra. Nothing in this package imports `torch` at module load time outside
the `smolvla`-specific code paths, so the light environment stays usable for
everything that does not need real model weights.

**Table 2.** The two environments.

| Environment | Contents | Use for | Install |
|---|---|---|---|
| `.venv` (light) | This package, `episodic-data-engine` core, `pydantic`, `pyyaml`, `typer`, `rich`, `psutil`, `matplotlib`, `numpy`. No `torch`. | Config validation, dataset profiling, interventions, tracking, reports, `mean_action`/`nearest_neighbor` baselines, and the CLI for everything that does not fine-tune SmolVLA. | `uv pip install --python ../.venv/bin/python -e .` |
| `.venv-vla` (full) | Everything above plus `torch` and `lerobot[smolvla]`, pulled in transitively through `episodic-data-engine[vla]`. | Real SmolVLA fine-tuning, inference, and any command that touches a checkpoint. | `uv pip install --python ../.venv-vla/bin/python -e .` |

Both commands are run from the `vla-finetune/` directory; `episodic-data-engine`
is installed editable via the `[tool.uv.sources]` entry in `pyproject.toml`, so
changes to either repository are picked up without reinstalling.

**Environment variables.**

| Variable | Effect | Default |
|---|---|---|
| `RDE_DATA_ROOT` | Overrides where `rde.config.get_settings()` resolves the data root (catalog, blobs, artifacts). Must be set consistently for every `mtrain`/`rde` invocation against the same data. | `rde.config`'s own default (a `rde_data/` sibling directory) |
| `RDE_REPO_PATH` | Overrides `model_training.experiments.runner.RDE_REPO`, used only for the git-provenance snapshot written into each run (`git.json`). | `../episodic-data-engine` relative to this package |
| `MODEL_TRAINING_REPO_PATH` | Overrides `model_training.experiments.runner.MT_REPO`, which also fixes the default `runs/` and `experiments/` directories when `ExperimentRunner` is constructed with no explicit `runs_dir`/`index_dir`. | This package's own repository root |

Every command below assumes `RDE_DATA_ROOT` is exported to the shared data
root before invocation, e.g.:

```bash
export RDE_DATA_ROOT=~/VLAExperiments/rde_data
../.venv-vla/bin/mtrain train --config configs/experiments/baseline.yaml
```

## 3. System Overview

`model_training` is a consumer of `episodic-data-engine`, never a modification
to it. The pipeline a training run walks is:

```
Dataset Version → DatasetView → ModelInputAdapter → Model → Prediction → Evaluation → Feedback
```

`rde` owns everything left of `DatasetView` (ingestion, storage, QC,
versioning) and the `Evaluator`/`FeedbackLoop` machinery on the right.
`model_training` owns: turning a declarative `ExperimentConfig` into a real,
lineage-tracked dataset version; orchestrating `SmolVLAPolicy.fit`; scoring the
result against real baselines; and packaging everything into a reproducible
run directory. The full design rationale for this split lives in
`ARCHITECTURE.md`; this document only needs the shape of the pipeline to make
sense of the API surface that follows.

## 4. Command-Line Interface (`mtrain`)

`mtrain` is a `typer` application (console script `model_training.cli:main`).
Every subcommand below is invoked as `mtrain <command> [options]`; run
`mtrain --help` or `mtrain <command> --help` for the same information from the
tool itself.

**Table 3.** Command index.

| Command | Purpose | Requires `.venv-vla` |
|---|---|---|
| `inspect` | Dataset overview, passthrough to `rde`'s own explorer. | No |
| `profile` | Build and print (optionally save) a full dataset profile. | No |
| `train` | Run one experiment end to end: intervention → train → evaluate → report. | Yes, if `training.model == "smolvla"` |
| `validate` | Print the selected checkpoint's validation metrics for a completed run. | No |
| `evaluate` | Re-run stratified + baseline offline evaluation for an existing run, against any split. | Yes |
| `test` | Alias for `evaluate --split test`. | Yes |
| `rollout` | Records that closed-loop rollout evaluation is unavailable in this environment. | No |
| `compare` | Model × dataset comparison table across several completed runs. | No |
| `analyze-failures` | Print a completed run's feedback artifact. | No |
| `export` | Copy a run's traceable artifacts into a `paper_ready/`-style export directory. | No |
| `build-intervention` | Resolve an `imbalanced`/`rebalanced` intervention against a real dataset profile and write it into a target config. | No |

### 4.1 `mtrain inspect`

```bash
mtrain inspect --dataset so100_pickplace --version training/v6
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `--dataset` | Yes | — | Dataset id. |
| `--version` | No | `training/v1` | Version string, e.g. `clean/v1`, `training/v6`. |

Prints `rde.exploration.queries.DatasetExplorer.overview(...)` as JSON.

### 4.2 `mtrain profile`

```bash
mtrain profile --dataset so100_pickplace --version training/v6 --out /tmp/profile.json
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `--dataset` | Yes | — | Dataset id. |
| `--version` | Yes | — | Version string. |
| `--out` | No | — | If given, also writes the profile JSON to this path. |

Builds and prints a `model_training.data.DatasetProfile` (Section 6.4).

### 4.3 `mtrain train`

```bash
mtrain train --config configs/experiments/baseline.yaml
mtrain train --config configs/experiments/reduced.yaml --seed 1
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `--config` | Yes | — | Path to an experiment YAML (Section 5). |
| `--seed` | No | The config's `seeds[0]` | Overrides the run seed without editing the file. |

Loads the config, runs `ExperimentRunner().run(cfg, seed=seed)`, and prints
the run directory and its `summary.json`. This is the command that performs
real gradient-based fine-tuning when `training.model == "smolvla"`; it
therefore needs `.venv-vla`. See Section 9.1 for the full workflow and
Section 8 for what it writes.

### 4.4 `mtrain validate`

```bash
mtrain validate --run runs/2026-09-03/baseline_full/seed_0
```

| Option | Required | Meaning |
|---|---|---|
| `--run` | Yes | Path to a completed run directory. |

Prints the selected checkpoint, the selection metric, its value, and the full
validation history (`metrics/validation.jsonl`). Exits with an error if the
run has no `training_outcome.json` (i.e. it never fine-tuned anything).

### 4.5 `mtrain evaluate` / `mtrain test`

```bash
mtrain evaluate --run runs/2026-09-03/baseline_full/seed_0 --split val
mtrain test --run runs/2026-09-03/baseline_full/seed_0        # == evaluate --split test
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `--run` | Yes | — | Path to a completed run directory. |
| `--split` | No (`evaluate` only) | `test` | Which split to score. |

Reloads the run's `selected_checkpoint` via `SmolVLAPolicy.from_checkpoint`
(no `contract` override — see Section 7.4) and re-runs
`stratified_evaluate(..., slices=["per_task"])` against the requested split,
**overwriting** `evaluation/test_overall.json` and `evaluation/stratified.json`
in that run directory. Use this to re-score an existing checkpoint after a
metrics-code change, or to score a split other than `test`.

### 4.6 `mtrain rollout`

```bash
mtrain rollout --run runs/2026-09-03/baseline_full/seed_0
```

Appends `{"status": "unavailable", "reason": "no simulator or physical robot connected in this environment"}`
to `metrics/rollout.jsonl` and prints it. There is no simulator or robot
connected in this environment; this command exists so that absence is a
recorded, honest fact rather than a silently missing file. It never
fabricates a rollout success rate.

### 4.7 `mtrain compare`

```bash
mtrain compare --runs runs/.../baseline_full/seed_0 --runs runs/.../reduced_data/seed_0 \
  --metric mse --out /tmp/comparison.csv
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `--runs` | Yes (repeatable) | — | One or more run directories. |
| `--metric` | No | `mse` | Metric column to compare (must exist in each run's `evaluation/test_overall.json`). |
| `--out` | No | — | If given, also writes the table as CSV. |

Prints a table with one row per run: model, dataset id, training version,
intervention kind, the requested metric for the model under test, and the
same metric for every baseline that run scored (columns named
`<baseline>_<metric>`, e.g. `mean_action_mse`).

### 4.8 `mtrain analyze-failures`

```bash
mtrain analyze-failures --run runs/2026-09-03/baseline_full/seed_0
```

Prints `failures/feedback_artifact.json` verbatim (Section 6.5). Exits with an
error if the run has no test-set model evaluation.

### 4.9 `mtrain export`

```bash
mtrain export --run runs/2026-09-03/baseline_full/seed_0 --output paper_ready/baseline_full
```

Copies a run's figures, evaluation JSON, resolved config, and manifests into
`<output>/{figures,tables,metrics,configs,manifests}/`, plus a generated
`experiment_summary.md` naming the source run and its `experiment_id`. Every
file in the export traces back to the source run directory; nothing is
recomputed or summarized lossily.

### 4.10 `mtrain build-intervention`

```bash
mtrain build-intervention \
  --dataset so100_pickplace --version training/v6 \
  --kind rebalanced --variable episode_length_bucket --variable-value short \
  --oversample-factor 2.0 \
  --target-config configs/experiments/rebalanced.yaml \
  --apply
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `--dataset` | Yes | — | Dataset id. |
| `--version` | Yes | — | Version whose per-episode metadata resolves the intervention. |
| `--kind` | Yes | — | `imbalanced` or `rebalanced` only — see below. |
| `--variable` | No | — | `task_id` or `episode_length_bucket`; the only two auto-resolvable grouping keys. |
| `--variable-value` | No | — | The value of `--variable` to target. |
| `--oversample-factor` | No | `3.0` | Per-episode sampling weight for `rebalanced` (see Section 5.2). |
| `--target-config` | Yes | — | YAML file to write the resolved `intervention:` block into. |
| `--reason` | No | `""` | Free-text justification recorded in the block and in lineage. |
| `--dry-run` / `--apply` | — | `--dry-run` | Without `--apply`, prints the resolved block and writes nothing. |

This command **does not build a dataset version** — `mtrain train` does that,
with real lineage, as part of a normal run. It only resolves *which* episodes
or weights an `imbalanced`/`rebalanced` intervention names, from the
dataset's actual per-episode metadata, and fails loudly rather than guessing
when `--variable` names something not resolvable. See Section 9.5.

## 5. Experiment Configuration Schema

Every experimental parameter that matters lives in one Pydantic object,
`model_training.config.ExperimentConfig`, loaded from YAML. It is the single
object a run is reproducible from, and the fully resolved instance (defaults
included) is what gets written as `config.resolved.yaml` inside the run
directory — a run never depends on an unwritten default.

### 5.1 File layout and merging

A config is normally two files: `configs/base.yaml` (shared defaults) and
`configs/experiments/<name>.yaml` (the experiment-specific override).
`model_training.config.load_experiment_config(path)` deep-merges the
experiment file onto the sibling `base.yaml` it finds two directories up from
`path`, then validates the merged result. An experiment file therefore only
needs to state what it changes:

```yaml
# configs/experiments/baseline.yaml
experiment_name: baseline_full
hypothesis: >
  SmolVLA fine-tuned on the complete, unmodified demonstration set
  establishes the reference point every intervention is compared against.
intervention:
  kind: full
```

Every other field — dataset, splits, training hyperparameters, evaluation
config — comes from `base.yaml` unless explicitly overridden.

### 5.2 `DatasetIntervention`

`kind` selects one of six mechanisms for what, if anything, changes about the
data before training. Table 4 lists every field; Table 5 lists which fields
each `kind` requires, and what mechanism actually executes it inside
`build_intervention_dataset` (Section 6.4).

**Table 4.** `DatasetIntervention` fields.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `kind` | `"full" \| "fraction" \| "reduced" \| "imbalanced" \| "rebalanced" \| "custom"` | `"full"` | Which mechanism applies. |
| `fraction` | `float \| None` (0–1) | `None` | Episode fraction for `fraction`/`reduced`. |
| `variable` | `str \| None` | `None` | Grouping key for `imbalanced`/`rebalanced`: `"task_id"` or `"episode_length_bucket"`. |
| `variable_value` | `Any` | `None` | The value of `variable` to target. |
| `episode_ids` | `list[int] \| None` | `None` | Explicit episode set; always wins over `variable` when given. |
| `episode_weights` | `dict[int, float] \| None` | `None` | Explicit per-episode sampling weight for `rebalanced`. |
| `oversample_factor` | `float` | `3.0` | Weight applied to the resolved target episodes when `episode_weights` is not given explicitly. |
| `seed` | `int` | `0` | Seed for the random draw (`fraction`/`reduced` only). |
| `reason` | `str` | `""` | Free-text justification, recorded in lineage. |

**Table 5.** Required fields and resolution mechanism, by `kind`.

| `kind` | Requires | Mechanism |
|---|---|---|
| `full` | nothing | No transform step; builds straight off the input version. |
| `fraction` | `fraction` or `episode_ids` | Seeded random draw of `fraction` of episodes, via `EpisodeSubsetFilter`, producing a real `clean/vN` with dropped-episode lineage. |
| `reduced` | nothing (`fraction` defaults to `0.5`) | Shorthand for `fraction` at the project-standard ~50%. |
| `imbalanced` | `variable` (+ `variable_value`) or `episode_ids` | The complement of the named group is kept — real episode exclusion, same `EpisodeSubsetFilter` mechanism. |
| `rebalanced` | `variable` (+ `variable_value`) or non-empty `episode_weights` | **No exclusion.** Every input episode is kept; the target episodes get a higher training *sampling weight* (`DatasetBuilder.build`'s `episode_weights`) — real data, oversampled in place, nothing duplicated or fabricated. |
| `custom` | `episode_ids` | Whatever the caller names explicitly. |

Leaving `variable` unset on `imbalanced`/`rebalanced` is deliberate schema
enforcement, not an oversight: the weak dimension must come from an actual
dataset profile, never be presumed in advance. Resolution fails loudly
(`ValueError`) rather than guessing when it cannot be resolved.

### 5.3 `SplitConfig`

Identical semantics to `rde.transformations.ops.assign_splits` — this *is*
that call, not a parallel schema that could disagree with it.

| Field | Type | Default |
|---|---|---|
| `train` | `float` | `0.8` |
| `val` | `float` | `0.1` |
| `test` | `float` | `0.1` |
| `seed` | `int` | `42` |
| `mode` | `"quantile" \| "independent"` | `"quantile"` |

`train + val + test` must sum to `1.0` (validated). Splits are always
episode-level, never frame-level.

### 5.4 `TrainingConfig`

Passed almost verbatim to `SmolVLAPolicy.fit` (Section 7.4).

**Table 6.** `TrainingConfig` fields.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `model` | `str` | `"smolvla"` | Only `"smolvla"` performs real gradient-based fine-tuning; anything else skips training and only scores baselines. |
| `pretrained_checkpoint` | `str` | `"lerobot/smolvla_base"` | HF Hub id or local path to start from. |
| `resume_from` | `str \| None` | `None` | Local checkpoint directory (one saved by `SmolVLAPolicy.save_checkpoint`) to resume from instead. See Section 9.2. |
| `steps` | `int` | `200` | Gradient steps this run performs. |
| `batch_size` | `int` | `4` | |
| `learning_rate` | `float` | `1e-4` | Peak LR (cosine schedule). |
| `weight_decay` | `float` | `1e-5` | |
| `grad_accum_steps` | `int` | `1` | |
| `max_grad_norm` | `float` | `1.0` | Gradient-clipping threshold. |
| `warmup_steps` | `int` | `0` | Linear warmup before the cosine decay. |
| `checkpoint_every` | `int` | `50` | Steps between checkpoint saves. |
| `validate_every` | `int` | `50` | Steps between validation passes (always on `val`, never `test`). |
| `val_batches` | `int` | `8` | Batches per validation pass (`val_batches × batch_size` frames). |
| `keep_last_n_checkpoints` | `int` | `3` | Older periodic checkpoints beyond this are pruned; the best-by-validation checkpoint and the final one are always kept regardless. |
| `num_workers` | `int` | `0` | |
| `mixed_precision` | `bool` | `False` | |
| `device` | `str \| None` | `None` | `None` auto-selects `cuda` if available, else `cpu`. |
| `seed` | `int` | `0` | Training seed (sampling order, not the split seed). |

### 5.5 `EvaluationConfig`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `split` | `str` | `"test"` | Split scored for the run's headline test metrics. |
| `batch_size` | `int` | `32` | |
| `max_frames` | `int \| None` | `None` | Caps frames evaluated; `None` evaluates the whole split. |
| `success_tolerance` | `float` | `0.1` | Tolerance band defining `Evaluator`'s `success_rate` proxy. |
| `baseline_models` | `list[str]` | `["mean_action", "nearest_neighbor"]` | Names from `MODEL_REGISTRY` (Section 7.5) scored alongside the model under test. |
| `selection_metric` | `str` | `"mse"` | Metric used for checkpoint selection — computed on `val`, never `test`. |
| `selection_mode` | `"min" \| "max"` | `"min"` | |
| `slices` | `list[str]` | `["per_task"]` | Stratified slices to additionally score at test time; see `stratified_evaluate` (Section 6.3). |

### 5.6 `ExperimentConfig` (top level)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `experiment_name` | `str` | — | Required; also the run directory's second path component. |
| `hypothesis` | `str` | `""` | What this run is meant to test. |
| `objective` | `str` | `""` | What this run establishes. |
| `dataset` | `DatasetRef` | — | `{dataset_id, version}` — the *input* version the intervention starts from. |
| `intervention` | `DatasetIntervention` | `full` | Section 5.2. |
| `splits` | `SplitConfig` | defaults above | Section 5.3. |
| `training` | `TrainingConfig` | defaults above | Section 5.4. |
| `evaluation` | `EvaluationConfig` | defaults above | Section 5.5. |
| `seeds` | `list[int]` | `[0]` | `seeds[0]` is used unless `--seed`/`seed=` overrides it. |
| `tags` | `dict[str, str]` | `{}` | Free-form; also indexed in `experiments/experiments.jsonl`. |
| `experiment_id` | `str \| None` | stamped at load | `sha256("{name}:{dataset_id}:{version}:{created_at}")[:10]`, prefixed with `experiment_name`. Not user-editable. |
| `created_at` | `str \| None` | stamped at load | UTC ISO-8601. |
| `source_config_path` | `str \| None` | stamped at load | The experiment YAML path passed to `load_experiment_config`. |

## 6. Python API — `model_training`

Everything below is importable from the module path shown; the same names
are re-exported from each submodule's `__init__.py`, so
`from model_training.experiments import ExperimentRunner` and
`from model_training.experiments.runner import ExperimentRunner` are
equivalent.

### 6.1 `model_training.config`

| Name | Signature | Purpose |
|---|---|---|
| `load_experiment_config` | `(path: str \| Path, base_path: str \| Path \| None = None) -> ExperimentConfig` | Load, deep-merge onto `base.yaml`, and validate one experiment config. |
| `load_yaml` | `(path: str \| Path) -> dict[str, Any]` | Plain YAML-to-dict, used internally and available standalone. |
| `ExperimentConfig`, `DatasetRef`, `DatasetIntervention`, `SplitConfig`, `TrainingConfig`, `EvaluationConfig` | Pydantic models | Section 5. |

### 6.2 `model_training.experiments`

**Table 7.** `ExperimentRunner`.

| Member | Signature | Notes |
|---|---|---|
| `__init__` | `(settings: Settings \| None = None, catalog: Catalog \| None = None, runs_dir: Path \| None = None, index_dir: Path \| None = None)` | Defaults: `get_settings()`, a fresh `Catalog`, `MT_REPO/runs`, `MT_REPO/experiments`. |
| `run` | `(config: ExperimentConfig, seed: int \| None = None) -> ExperimentRunResult` | The full lifecycle — see Section 6.2.1. |

`ExperimentRunResult` is a `dataclass` with `run_dir: Path`,
`experiment_id: str`, `training_version: str`, `summary: dict[str, Any]`
(the same object written as `summary.json`).

#### 6.2.1 What `ExperimentRunner.run` does

In order, every step below is a real call into `rde` or a packaging of that
call's real output — nothing is fabricated, and a step that cannot execute
(no GPU, no simulator) records why rather than being silently skipped:

1. Create the run directory (`make_run_dir`, Section 6.7) and write
   `environment.json`, `git.json`, `config.resolved.yaml`.
2. Materialize the intervention as a real dataset version
   (`build_intervention_dataset`, Section 6.4) and write `dataset.json`.
3. Open a `DatasetView` on the resulting `training/vN`, verify split
   integrity (`verify_split_integrity`, Section 6.4.3), and write
   `split_integrity.json`.
4. Build and write `dataset_profile.json` plus two figures (episode-length and
   task distribution).
5. If `training.model == "smolvla"`: run `Trainer.train` (Section 6.3),
   write `training_outcome.json`, and render the training-curve and
   validation-curve figures. Otherwise record a limitation and skip to step 7
   with no selected checkpoint.
6. Write `model.json` naming the selected checkpoint.
7. Score every configured baseline (`evaluate_baselines`) and write
   `evaluation/baselines.json`.
8. If a checkpoint was selected: reload it (`SmolVLAPolicy.from_checkpoint`),
   run `stratified_evaluate` against the configured split and slices, write
   `evaluation/test_overall.json`, `evaluation/stratified.json`, an
   error-distribution figure, and `predictions/test_outcomes.jsonl`; build and
   register a `ModelArtifact`; build and write the feedback artifact
   (`failures/feedback_artifact.json`); render `reports/model_card.md`.
9. Append `metrics/test.jsonl` and `metrics/rollout.jsonl` (the latter always
   `"unavailable"` in this environment).
10. Render `reports/experiment_report.md`.
11. Write `summary.json` and record the run, its dataset, and (if scored) its
    metrics into the four `experiments/*.jsonl` index files.

**Table 8.** `run_data_fraction_sweep` (`model_training.experiments.data_fraction`).

| Signature | `(base_config: ExperimentConfig, fractions: list[float], runner: ExperimentRunner \| None = None) -> list[dict[str, Any]]` |
|---|---|
| Purpose | Runs `base_config` once per fraction, varying only `intervention = {kind: "fraction", fraction: f}` — every other knob (model, hyperparameters, splits, seed) held fixed, isolating the quantity effect. Each fraction gets its own `experiment_name` (`{base}_frac{pct}`) and a fresh `experiment_id`. Returns one summary dict per fraction (`{"fraction": f, **run_summary}`). |
| Note | Fractions are **not** run unless this function is explicitly called — it makes the sweep possible, not automatic. |

### 6.3 `model_training.training`

**Table 9.** `Trainer`.

| Member | Signature | Notes |
|---|---|---|
| `__init__` | `(run_dir: RunDir, config: ExperimentConfig)` | |
| `train` | `(train_view: DatasetView, val_view: DatasetView, *, episode_ids: list[int] \| None = None, episode_weights: dict[int, float] \| None = None) -> TrainingOutcome` | Raises `ValueError` if `config.training.model != "smolvla"`. |

`train` constructs a `SmolVLAPolicy` (resuming from `training.resume_from` if
set, otherwise from `training.pretrained_checkpoint`), calls `.load()` with
the training dataset's contract when starting fresh (Section 7.4), then calls
`policy.fit(...)` inside a `try`/`except`, with an `on_step` callback that
appends `metrics/train.jsonl`, and — every `checkpoint_every`/`validate_every`
steps — runs a validation pass (on `val`, never `test`), saves a checkpoint
under `checkpoints/step_<n>/`, immediately verifies it via
`SmolVLAPolicy.verify_checkpoint` (a failed verification is logged to
`metrics/checkpoint_errors.jsonl` and marks that checkpoint `"verified":
false`, ineligible to become `best`), and prunes older periodic checkpoints
beyond `keep_last_n_checkpoints` (always keeping the running best and, at the
end, `checkpoints/final/`).

If `policy.fit(...)` raises (OOM, a NaN loss, a signal), `Trainer.train`
writes `crash.json` to the run directory — `error_type`, `error`,
`steps_completed`, `checkpoints` recorded so far, `wall_clock_s` — releases
GPU memory (`del policy; torch.cuda.empty_cache()`), and re-raises. No final
checkpoint is written on this path; `checkpoints/final/` is only produced
after `policy.fit()` returns normally. On a normal return, `Trainer.train`
also releases GPU memory before returning, since `ExperimentRunner.run`
constructs a second `SmolVLAPolicy` for evaluation immediately afterward.

`TrainingOutcome` (`dataclass`): `steps_completed: int`,
`final_train_loss: float | None`, `best_checkpoint: str | None`,
`best_val_metric: float | None`, `selection_metric: str`,
`checkpoints: list[dict]` (one entry per save, with `step`, `path`,
`val_metric`, `verified`, and `pruned`/`final` flags where applicable),
`peak_rss_mb`, `peak_gpu_mb`, `wall_clock_s`.

### 6.4 `model_training.data`

**Table 10.** `model_training.data.interventions`.

| Name | Signature | Purpose |
|---|---|---|
| `build_intervention_dataset` | `(dataset_id: str, input_version: str, intervention: DatasetIntervention, splits: SplitConfig, settings=None, catalog=None) -> InterventionResult` | Turns one `DatasetIntervention` into a real, lineage-tracked `clean/vN` (if exclusion is needed) plus a real `training/vN`. See Section 9.5 for the underlying mechanism. |
| `resolve_episode_subset` | `(intervention, all_episode_ids, task_of_episode, length_bucket_of_episode=None) -> tuple[list[int], str]` | Episode ids to keep, and the reason, for every `kind` except `rebalanced`. An explicit `episode_ids` always wins. |
| `compute_length_bucket` | `(episode_rows: list[dict]) -> dict[int, str]` | short/medium/long by within-dataset episode-length terciles — the always-available grouping key regardless of task diversity. |

`InterventionResult` (`dataclass`): `dataset_id`, `input_version`,
`clean_version`, `training_version`, `kind`, `episode_ids`,
`excluded_episode_ids`, `episode_weights`, `selection_reason`,
`build_summary`.

**Table 11.** `model_training.data.profile`.

| Name | Signature | Purpose |
|---|---|---|
| `build_dataset_profile` | `(dataset_id: str, version: str, settings=None, catalog=None) -> DatasetProfile` | Read-only snapshot of one dataset version: overview, duration distribution, task/modality summary, per-dimension state/action statistics, QC summary, lineage, split distribution, and coverage (normalized Shannon entropy per numeric dimension — 0 = one bin, 1 = uniform). |

`DatasetProfile.to_dict()` is the exact shape written as
`dataset_profile.json` (fields: `dataset_id`, `version`, `parent_version`,
`generated_at`, `episode_count`, `frame_count`, `task_count`, `tasks`,
`episode_length`, `action`, `state`, `modalities`, `image`, `language`, `qc`,
`splits`, `missing_data`, `duplicate_summary`, `coverage`, `notes`). Anything
not honestly computable from the dataset's real schema (e.g. visual
diversity, which needs an embedding model this profiler does not run) is
reported as `"unavailable"` with a reason, never approximated silently.

**Table 12.** `model_training.data.split_integrity`.

| Name | Signature | Purpose |
|---|---|---|
| `verify_split_integrity` | `(view: DatasetView, *, raise_on_failure: bool = True) -> SplitIntegrityReport` | Independently re-derives train/val/test disjointness from the materialized `DatasetView`, rather than trusting the code path that built it. Raises `SplitLeakageError` on any overlap when `raise_on_failure` (the default). |

### 6.5 `model_training.feedback`

| Name | Signature | Purpose |
|---|---|---|
| `build_feedback_artifact` | `(experiment_id, model_id, dataset_id, dataset_version, result: EvaluationResult, *, settings=None, catalog=None, error_percentile: float = 90.0, min_failures_per_episode: int = 3) -> dict` | Wraps `rde.feedback.FeedbackLoop` (real failure detection, never reimplemented) and adds candidate explanations (concentrated vs. scattered failure — a model failure never automatically implies bad data) and a concrete `recommended_next_action`. |

Return shape: `experiment_id`, `model_id`, `dataset_id`, `dataset_version`,
`affected_slice`, `failure_summary`, `ranked_episode_reviews`,
`candidate_explanations`, `recommended_next_action`, `retraining_selection`,
`rde_artifacts`. Full methodology in `FAILURE_ANALYSIS.md`.

### 6.6 `model_training.evaluation`

All five functions build on `rde.evaluation.Evaluator` (Section 7.3); none
duplicate its logic.

**Table 13.** `model_training.evaluation`.

| Name | Signature | Purpose |
|---|---|---|
| `evaluate_baselines` | `(view, names: list[str], *, split="test", train_split="train", batch_size=32, max_frames=None) -> dict[str, EvaluationResult]` | Fits and scores each named baseline (from `MODEL_REGISTRY`) through the same evaluation call used for the model under test. |
| `stratified_evaluate` | `(view, model, adapter, *, base_split="test", batch_size=32, max_frames=None, slices=None) -> dict[str, EvaluationResult]` | `"overall"` plus one entry per requested slice. Supported slice names: `"per_task"` (one per distinct `task_id`), `"per_length_bucket"` (short/medium/long by within-split terciles, `"length_short"`/`"length_medium"`/`"length_long"` keys). Any other name raises `ValueError` — nothing is silently approximated. |
| `error_distribution` | `(result: EvaluationResult) -> dict` | `median`/`p90`/`p95`/`p99`/`min`/`max` of per-frame error. |
| `episode_metrics` | `(result: EvaluationResult) -> list[dict]` | Per-episode `mae`/`mse`/`rmse`/`max_error`, sorted by episode id. |
| `temporal_error` | `(result: EvaluationResult, view: DatasetView, buckets: int = 10) -> dict` | Mean error vs. normalized within-episode progress (10 buckets by default). |

### 6.7 `model_training.tracking`

**Table 14.** `RunDir` and helpers.

| Name | Signature | Purpose |
|---|---|---|
| `make_run_dir` | `(base_runs_dir: Path, experiment_name: str, seed: int, date: str \| None = None) -> RunDir` | Creates `<base>/<date>/<experiment_name>/seed_<seed>/` and its subdirectories. **Raises `FileExistsError` if that path already exists and is non-empty** — a run directory is never silently overwritten; a same-day rerun needs a different `experiment_name` or its old directory cleared first. |
| `RunDir.path` | property `-> Path` | The run's root directory. |
| `RunDir.write_json` | `(name: str, data: Any) -> Path` | |
| `RunDir.append_jsonl` | `(relpath: str, record: dict) -> None` | |
| `RunDir.sub` | `(name: str) -> Path` | Returns (creating if needed) a subdirectory. |
| `collect_environment` | `() -> dict` | Python/OS/CPU/RAM, plus `torch`/`cuda`/`gpu`/`lerobot` versions where installed — `None` with no fabricated value where a dependency is absent. |
| `collect_git_info` | `(rde_repo: Path, model_training_repo: Path) -> dict` | Commit, dirty flag, and branch for both repositories a run's code spans. |

**Table 15.** `ExperimentIndex` — four append-only JSONL files at
`<project>/experiments/`, each queryable independently with `jq`,
`pandas.read_json(lines=True)`, or DuckDB, without this project running.

| File | Written by | One row per |
|---|---|---|
| `experiments.jsonl` | `record_experiment(**fields)` | completed run |
| `metrics.jsonl` | `record_metric(**fields)` | (run, split, metric snapshot) |
| `datasets.jsonl` | `record_dataset(**fields)` | dataset version an experiment touched |
| `models.jsonl` | `record_model(**fields)` | registered model artifact |

`JsonlIndex` (the primitive behind each file) also exposes
`read_all() -> list[dict]` and `query(**filters) -> Iterator[dict]`.

### 6.8 `model_training.deployment`

**Table 16.** `model_training.deployment`.

| Name | Signature | Purpose |
|---|---|---|
| `build_model_artifact` | `(*, checkpoint_path, contract, run_dir_path, experiment_id, tags, dataset_id, dataset_version, environment, git_info, metrics) -> ModelArtifact` | Assembles a `ModelArtifact` from an already-saved checkpoint (verified via `SmolVLAPolicy.verify_checkpoint`, not assumed complete). |
| `DeployedModel` | class | Section 6.8.1. |
| `ModelRegistry` | class | Section 6.8.2. |

`ModelArtifact` (`dataclass`): `model_id`, `model_version`, `artifact_path`,
`architecture`, `pretrained_model`, `training_run_id`, `experiment_id`,
`dataset_id`, `dataset_version`, `observation_schema`, `action_schema`,
`embodiment`, `dependencies`, `git_commit`, `created_at`, `metrics`.

#### 6.8.1 `DeployedModel`

The full deployment path — saved artifact → deterministic loader →
preprocessing → SmolVLA → action → metrics — as a plain sequence of function
calls, no service/queue/orchestration layer. See `DEPLOYMENT.md` for the
design rationale.

| Member | Signature |
|---|---|
| `__init__` | `(checkpoint_path: str, contract: dict, device: str \| None = None)` |
| `health_check` | `() -> dict` — verifies checkpoint completeness, a valid action shape, at least one image modality with a valid `(H, W, C)` shape, and normalization stats on every inline state modality. Raises `DeploymentHealthError` naming the specific failure; never falls back to a different model. |
| `load` | `() -> DeployedModel` — calls `health_check()` first, then `SmolVLAPolicy.from_checkpoint`. |
| `predict` | `(samples: Sequence[Sample]) -> tuple[Prediction, LatencyBreakdown]` | |

`LatencyBreakdown` (`dataclass`): `model_load_ms`, `preprocessing_ms`
(image decode + resize + tensor construction, one bucket — not separately
instrumented), `inference_ms`, `postprocessing_ms`, `total_ms`, `note`.

#### 6.8.2 `ModelRegistry`

A local registry over the same `JsonlIndex` mechanism (`experiments/models.jsonl`).

| Member | Signature |
|---|---|
| `__init__` | `(path: Path)` |
| `register` | `(artifact: ModelArtifact) -> None` |
| `get` | `(model_id: str) -> dict \| None` |
| `list` | `() -> list[dict]` |
| `for_experiment` | `(experiment_id: str) -> list[dict]` |

**Not wired into `ExperimentRunner`.** `ExperimentRunner.run` records each
model directly via `ExperimentIndex.record_model` (Section 6.7, Table 15) —
it never calls `ModelRegistry.register`. `ModelRegistry` is tested in
isolation but is not the live registration path. Because the backing file is
append-only, a `model_id` registered more than once has several rows; `get`
returns the most recent of them, so later registrations supersede earlier
ones.

### 6.9 `model_training.reports`

**Table 17.** `model_training.reports`.

| Name | Signature | Purpose |
|---|---|---|
| `load_run_summary` | `(run_dir: str \| Path) -> dict` | Reads a run's `dataset.json`/`model.json`/`manifest.json`/`summary.json`/evaluation files back from disk — never live training state, so a comparison is reproducible from a previous session. |
| `build_comparison_table` | `(run_dirs: list[str \| Path], metric: str = "mse") -> list[dict]` | One row per run: model, dataset id, training version, intervention, the requested metric for the model under test and for each baseline it scored. |
| `write_comparison_csv` | `(rows: list[dict], out_path: str \| Path) -> Path` | |
| `build_model_dataset_matrix` | `(rows: list[dict], metric: str = "mse") -> list[dict]` | The model × dataset/intervention summary table. |
| `render_experiment_report` | `(ctx: dict) -> str` | Renders `reports/experiment_report.md`'s Markdown from the context `ExperimentRunner.run` assembles (Section 6.2.1, step 10). |
| `render_model_card` | `(artifact: dict, ctx: dict) -> str` | Renders `reports/model_card.md`. |
| `figures.plot_training_curves` | `(train_records: list[dict], out_dir: Path) -> list[Path]` | Loss, gradient-norm, and learning-rate curves from `metrics/train.jsonl`. |
| `figures.plot_validation_curve` | `(val_records: list[dict], out_dir: Path, metric: str = "mse") -> Path \| None` | |
| `figures.plot_error_distribution` | `(error_dist: dict, out_dir: Path) -> Path \| None` | |
| `figures.plot_episode_length_distribution` | `(profile: dict, out_dir: Path) -> Path \| None` | |
| `figures.plot_task_distribution` | `(profile: dict, out_dir: Path) -> Path \| None` | |
| `figures.plot_model_comparison` | `(rows: list[dict], metric: str, out_dir: Path, ...) -> Path \| None` | |
| `figures.plot_data_fraction_curve` | `(rows: list[dict], metric: str, out_dir: Path) -> Path \| None` | |

## 7. Python API — `episodic-data-engine` (consumed directly)

Training and evaluation code written against this project calls a handful of
`rde` classes directly, in addition to going through `model_training`'s own
wrappers. This section documents exactly that subset — not `rde` in general;
see `episodic-data-engine/docs/` for the platform's own reference.

### 7.1 `DataEngine` and `DatasetView`

```python
from rde.config import get_settings
from rde.engine import DataEngine

engine = DataEngine(get_settings())
view = engine.get_dataset("so100_pickplace", "training/v6")
try:
    for sample in view.iter_samples(split="train"):
        ...
finally:
    view.close()
```

`DataEngine(settings: Settings | None = None)` exposes `list_datasets()`,
`list_versions(dataset_id=None)`, `latest_version(dataset_id, stage)`, and
`get_dataset(dataset_id, version="training/v1") -> DatasetView`.

**Table 18.** `DatasetView` — a read-only handle on one dataset version.
Consumers never learn where the bytes live or what ingested/transformed them.

| Member | Signature | Notes |
|---|---|---|
| `dataset_id`, `version`, `fps` | properties | |
| `contract` | property `-> dict` | The model-facing training contract: `{"observation": {name: {key, kind, dtype, shape, plane, fps, normalization, ...}}, "action": {...}}`. Every `ModelInputAdapter` reads this instead of hard-coded key names. |
| `episode_ids` | `(split: str \| None = None, task_id: int \| None = None) -> list[int]` | |
| `episode` | `(episode_id: int) -> Episode \| None` | |
| `splits` | `() -> dict[str, int]` | Episode counts per split; an `"unassigned"` bucket means no split has been built yet (a `clean/vN` version, pre-`DatasetBuilder.build`). |
| `iter_samples` | `(split=None, episode_ids=None, batch_size=None) -> Iterator[Sample]` | Images stay undecoded (`Sample.image_refs`) until `Sample.image(modality)` is called. |
| `iter_frames` | `(split=None, episode_ids=None, columns=None, batch_size=None) -> Iterator[pa.RecordBatch]` | Lower-level, columnar. |
| `get_episode_frames` | `(episode_id: int, columns: Sequence[str] \| None = None) -> pa.Table` | Whole-episode read, bounded by episode length. |
| `close` | `() -> None` | Also usable as a context manager. |

`Sample` (`dataclass`): `frame_id`, `episode_id`, `frame_index`,
`timestamp_s`, `task_id`, `task`, `observation: dict[str, np.ndarray]`,
`action: np.ndarray | None`, plus `.image(modality) -> np.ndarray | None`
(lazy decode) and `.image_modalities -> list[str]`.

### 7.2 `Model` / `ModelInputAdapter` / `ModelInput` / `Prediction`

The seam that keeps the data platform independent of any model
(`rde.model_adapters.base`). Adapters import from the engine; the engine
never imports from a model.

```python
class ModelInputAdapter(ABC):
    def __init__(self, contract: dict): ...        # reads action_key, action_dim,
                                                      # image_keys (plane == "external"),
                                                      # state_keys (plane == "inline")
    @abstractmethod
    def adapt(self, samples: Sequence[Sample]) -> ModelInput: ...

class Model(ABC):
    def fit(self, view: DatasetView, split: str = "train", **kwargs) -> None: ...  # optional
    @abstractmethod
    def predict(self, model_input: ModelInput) -> Prediction: ...
    def timed_predict(self, model_input: ModelInput) -> Prediction: ...  # wraps predict() with latency
```

`ModelInput` (`dataclass`): `tensors: dict[str, np.ndarray]`,
`instruction: str | None`, `frame_ids`, `episode_ids`, `metadata`.

`Prediction` (`dataclass`): `actions: np.ndarray` `(batch, action_dim)`,
`confidence: np.ndarray` `(batch,)` in `[0, 1]` — **required**, not optional:
the feedback loop selects low-confidence examples for review, so a model
without a native uncertainty notion must derive one and say how in
`metadata` — plus `frame_ids`, `episode_ids`, `latency_ms`, `model_name`,
`metadata`.

Three concrete adapters exist in `rde.model_adapters.adapters`:
`StateAdapter` (state only — concatenates **every** inline modality present
in the contract, in contract order, into one `(batch, dim)` array; see the
warning in Section 9.6), `VisionStateAdapter` (state + one decoded tensor per
camera, `NCHW` by default), and `VisionLanguageAdapter` (`VisionStateAdapter`
plus the per-sample task instruction — what `SmolVLAPolicy` uses).

### 7.3 `Evaluator`

```python
from rde.evaluation import Evaluator

result = Evaluator().evaluate(
    view, model, adapter, split="test", batch_size=32,
    max_frames=None, success_tolerance=0.1, persist=True, episode_ids=None,
)
```

`evaluate(...) -> EvaluationResult` runs `model` over one split (or an
explicit `episode_ids` set, which takes precedence over `split` for frame
selection — `split` is still recorded and used to name persisted artifacts,
so callers evaluating a slice should pass a descriptive label such as
`"test:task_pickplace"`). `EvaluationResult` carries `.metrics` (`mse`, `mae`,
`rmse`, `per_dimension_mae`, `normalized_mae`, `success_rate`,
`mean_confidence`, `confidence_error_corr`, `latency_ms_per_frame`,
`action_dim`), `.summary()`, `.outcomes` (per-frame `FrameOutcome` records:
`frame_id`, `episode_id`, `error`, `confidence`, `per_dim_error`), `.split`,
`.frames_evaluated`. Full metric definitions in `METRICS.md`.

### 7.4 `SmolVLAPolicy`

`rde.model_adapters.models.SmolVLAPolicy` — the adapter for the real SmolVLA
checkpoint.

**Table 19.** `SmolVLAPolicy` public surface.

| Member | Signature | Notes |
|---|---|---|
| `__init__` | `(checkpoint: str = "lerobot/smolvla_base", device: str \| None = None)` | |
| `load` | `(contract: dict \| None = None) -> SmolVLAPolicy` | Section 7.4.1. |
| `fit` | `(view, split="train", *, steps=200, batch_size=4, learning_rate=1e-4, weight_decay=1e-5, grad_accum_steps=1, max_grad_norm=1.0, warmup_steps=0, episode_ids=None, episode_weights=None, seed=0, optimizer_state=None, on_step=None, **kwargs) -> dict` | Real gradient-descent fine-tuning. Returns `{"steps", "final_loss", "history", "optimizer_state"}`. |
| `predict` | `(model_input: ModelInput) -> Prediction` | Uses `predict_action_chunk` (a fresh forward pass every call), **not** `select_action` (which maintains a stateful action queue meant for one continuous rollout — reusing it across independent evaluation batches silently replays stale actions). |
| `save_checkpoint` | `(path: str \| Path, *, extra: dict \| None = None) -> dict` | Saves weights **and** the exact pre/post-processor pipelines used to train them — required, because normalization statistics and the language tokenizer live in the processors, not the weights. |
| `verify_checkpoint` | `(path) -> dict` (staticmethod) | Raises `FileNotFoundError`/`ValueError` naming the specific missing file on an incomplete/foreign directory; no partial load is ever attempted. |
| `from_checkpoint` | `(path, device=None) -> SmolVLAPolicy` (classmethod) | Verifies, then loads with `contract=None` — a checkpoint saved by `save_checkpoint` is already correctly configured for whatever dataset it was trained on. |
| `train_mode` / `eval_mode` | `() -> None` | Flips the underlying `torch.nn.Module` without reaching into `_policy`. |
| `describe` | `() -> dict` | `{"model", "type", "checkpoint", "device", "loaded"}`. |

#### 7.4.1 `load(contract=None)` and cross-embodiment support

`lerobot/smolvla_base` ships pre-configured for whatever embodiment it was
itself tutorial-trained on (`so100_pickplace`: 6-DoF state/action, 2
cameras). Loading it as-is against a *different* dataset silently keeps that
embodiment's shapes — wrong, not merely suboptimal, since a 14-DoF bimanual
action forced through a 6-wide head cannot be represented at all.

Passing `contract` (a `DatasetView.contract`, Section 7.1) overrides both the
checkpoint's declared `input_features`/`output_features` and its
normalization statistics before weights load:

- **Shapes.** SmolVLA pads state/action internally to a fixed
  `max_state_dim`/`max_action_dim` budget (32 in the current checkpoint)
  regardless of the real per-dataset width, so declaring the real width
  changes what the model treats as signal vs. padding — it does not resize
  any weight tensor. Camera count is unconstrained the same way: each camera
  goes through the same shared vision encoder, so more or fewer cameras only
  changes how many times that encoder runs per forward pass.
- **Statistics.** The checkpoint's saved processor encodes *its own*
  embodiment's joint distribution. Reusing it for a different one would
  silently misnormalize every input, so a `contract`-driven load builds a
  fresh preprocessor from **this dataset's own ingested normalization
  statistics** instead of the checkpoint's saved ones.

`Trainer.train` passes `contract=None` when `training.resume_from` is set
(the checkpoint being resumed is already correctly configured for its
dataset) and `contract=train_view.contract` otherwise. See Section 9.6 for
the full onboarding workflow and its gotchas.

### 7.5 Model registry (`rde.model_adapters`)

| Name | Signature | Purpose |
|---|---|---|
| `MODEL_REGISTRY` | `dict[str, tuple[type, type[ModelInputAdapter]]]` | `{"mean_action": (MeanActionPolicy, StateAdapter), "nearest_neighbor": (NearestNeighborPolicy, StateAdapter), "smolvla": (SmolVLAPolicy, VisionLanguageAdapter)}`. |
| `build_model` | `(name: str, contract: dict, **kwargs) -> tuple[Model, ModelInputAdapter]` | Instantiates a registered model and its paired adapter together. |

`MeanActionPolicy` (state-independent floor baseline) and
`NearestNeighborPolicy` (behavior retrieval — nearest training state, replays
its action; CPU-only, a genuinely strong baseline on single-task data) are
the two reference baselines every real model is compared against.

## 8. Run Directory Layout

Every completed `mtrain train` run lives at
`runs/<date>/<experiment_name>/seed_<seed>/`, created once by `make_run_dir`
and never silently reused. Table 20 is the observed layout of a real,
completed run.

**Table 20.** Run directory contents.

| Path | Written by | Contents |
|---|---|---|
| `config.resolved.yaml` | `ExperimentRunner.run` | Every `ExperimentConfig` field, defaults included. |
| `environment.json` | `collect_environment` | Python/OS/hardware, `torch`/`cuda`/`gpu`/`lerobot` versions. |
| `git.json` | `collect_git_info` | Commit/dirty/branch for both repos, under keys `episodic_data_engine` and `vla_finetune`. |
| `dataset.json` | `ExperimentRunner.run` | Intervention result: input/clean/training versions, kept/excluded episode ids, weights, reason. |
| `dataset_profile.json` | `build_dataset_profile` | Full `DatasetProfile` (Section 6.4). |
| `split_integrity.json` | `verify_split_integrity` | Independently re-derived split disjointness. |
| `model.json` | `ExperimentRunner.run` | Architecture, pretrained checkpoint, selected checkpoint, selection metric/value. |
| `model_artifact.json` | `build_model_artifact` | Full `ModelArtifact` — the deployable identity of this run's checkpoint. |
| `training_outcome.json` | `Trainer.train` | Full `TrainingOutcome` (Section 6.3). |
| `summary.json` | `ExperimentRunner.run` | The compact, indexed summary — also what `mtrain train` prints. |
| `checkpoints/step_<n>/`, `checkpoints/final/` | `SmolVLAPolicy.save_checkpoint` | `config.json`, `model.safetensors`, `policy_preprocessor.json` (+ its `.safetensors`), `policy_postprocessor.json` (+ its `.safetensors`), `rde_checkpoint_meta.json`. |
| `metrics/train.jsonl` | `Trainer.train`'s `on_step` | One record per gradient step: `step`, `loss`, `lr`, `grad_norm`. |
| `metrics/validation.jsonl` | `Trainer.train`'s `on_step` | One record per validation pass: `step` + full `EvaluationResult.metrics`. |
| `metrics/checkpoint_errors.jsonl` | `Trainer.train`'s `on_step` | Present only if a checkpoint save failed `verify_checkpoint`: `step`, `path`, `error`. That checkpoint is recorded with `"verified": false` and is never eligible as `best`. |
| `crash.json` | `Trainer.train` | Present only if `policy.fit(...)` raised: `error_type`, `error`, `steps_completed`, `checkpoints` saved so far, `wall_clock_s`. No `checkpoints/final/` exists in this case. |
| `metrics/test.jsonl` | `ExperimentRunner.run` | One record: the test-set `EvaluationResult.metrics`. |
| `metrics/rollout.jsonl` | `mtrain rollout` / `ExperimentRunner.run` | `{"status": "unavailable", "reason": ...}` — never fabricated. |
| `evaluation/baselines.json` | `ExperimentRunner.run` | `{baseline_name: metrics}` for every configured baseline. |
| `evaluation/test_overall.json` | `ExperimentRunner.run` | `summary`, `metrics`, `error_distribution`, `episode_metrics`, `temporal_error` for the whole test split. |
| `evaluation/stratified.json` | `ExperimentRunner.run` | `{slice_name: {**metrics, "frames"}}` for every configured slice. |
| `predictions/test_outcomes.jsonl` | `ExperimentRunner.run` | Per-frame outcome rows: run/experiment/dataset identity, frame/episode id, error, confidence, per-dimension error — traceable back to source via `rde version trace`. |
| `failures/feedback_artifact.json` | `build_feedback_artifact` | Section 6.5. |
| `reports/experiment_report.md` | `render_experiment_report` | Full narrative report. |
| `reports/model_card.md` | `render_model_card` | Model card for the selected checkpoint. |
| `reports/figures/*.png` | `model_training.reports.figures` | Training loss, gradient norm, learning rate, validation curve, error distribution, episode-length distribution, task distribution. |
| `resume_source.json` | `Trainer.train` | Present only when `training.resume_from` was set — the resumed checkpoint's verified metadata. |

## 9. Common Workflows

### 9.1 Launch a new experiment

1. Write (or copy and edit) an experiment YAML under `configs/experiments/`
   — only fields that differ from `configs/base.yaml` are needed (Section
   5.1).
2. `mtrain train --config configs/experiments/<name>.yaml` (from `.venv-vla`
   if `training.model == "smolvla"`).
3. Inspect `runs/<date>/<name>/seed_<seed>/summary.json` and
   `reports/experiment_report.md`.

### 9.2 Resume or continue training from a checkpoint

Set `training.resume_from` to a checkpoint directory saved by
`SmolVLAPolicy.save_checkpoint` (e.g. a previous run's
`checkpoints/step_1000`) and give the run a new `experiment_name` (resuming
into the same run directory is not supported — `make_run_dir` refuses a
non-empty directory). Because the LR schedule restarts (warmup then cosine
decay across the *new* `steps` budget, not a continuation of the original
schedule), lower `learning_rate` and shorten `warmup_steps` relative to the
original run to avoid perturbing an already-converged model:

```yaml
training:
  resume_from: /abs/path/to/previous/run/checkpoints/step_1000
  steps: 1000
  learning_rate: 3.0e-5
  warmup_steps: 20
```

The new run's own `metrics/train.jsonl`/`validation.jsonl` step numbers start
at 0 again — they are steps *within this run*, not cumulative; add the
previous run's `steps_completed` when reporting a cumulative step count.

### 9.3 Re-evaluate an existing run

`mtrain evaluate --run <dir> --split val` (or `mtrain test --run <dir>` for
`test`) reloads the run's selected checkpoint and overwrites
`evaluation/test_overall.json`/`evaluation/stratified.json` in place — use
after a metrics-code change, or to score a split the original run did not.

### 9.4 Compare runs

`mtrain compare --runs <dir1> --runs <dir2> ... --metric mse --out table.csv`.
Reads only persisted artifacts (Section 6.9's `load_run_summary`), so this
works across sessions and does not require the runs to still be "live".

### 9.5 Build a reduced / imbalanced / rebalanced intervention

1. `mtrain build-intervention --dataset <id> --version <clean/vN> --kind imbalanced --variable episode_length_bucket --variable-value short --target-config configs/experiments/imbalanced.yaml` (omit `--apply` first to inspect the resolved block).
2. Re-run with `--apply` to write the resolved `intervention:` block into the
   target config.
3. `mtrain train --config configs/experiments/imbalanced.yaml` — this is what
   actually builds the dataset version, via `build_intervention_dataset`
   (Section 6.4), with real lineage.

`reduced` needs no `build-intervention` step (`kind: reduced` alone resolves
to a seeded ~50% draw). `rebalanced` never excludes episodes — it raises
their training sampling weight, real data oversampled in place.

### 9.6 Onboard a new dataset or embodiment

Worked example from this project: adding `aloha_coffee` (bimanual ALOHA,
14-DoF action, 4 cameras) alongside `so100_pickplace` (6-DoF, 2 cameras).

1. **Ingest and version the dataset through `rde`** (source ingestion, QC,
   `clean/vN`) — outside this package's scope; see
   `episodic-data-engine/docs/`.
2. **Write an experiment config** naming the new `dataset.dataset_id`/
   `version`; everything else (training hyperparameters, splits, evaluation)
   can inherit from `base.yaml` unchanged.
3. **Run a short dry run first** (a handful of steps, e.g. `training.steps: 8`
   in a throwaway experiment file) before committing to a full run. This is
   the cheapest way to surface a schema mismatch specific to the new
   dataset.
4. **Know what `Trainer.train` does differently for a fresh checkpoint**: it
   calls `SmolVLAPolicy.load(contract=train_view.contract)`, which
   reconfigures the checkpoint's input/output shapes and normalization
   statistics for the new dataset (Section 7.4.1) — no manual step needed,
   but two things it depends on being correct in the ingested contract are
   worth knowing about directly:
   - **State width equals every inline modality's width, summed.**
     `ModelInputAdapter`/`StateAdapter` concatenate *every* observation
     modality with `plane == "inline"` into one `"state"` tensor, not just
     the one named `state.joint`. A dataset with a second inline modality
     (e.g. `force.effort`) produces a wider state vector than
     `state.joint`'s own shape suggests — `_policy_features_from_contract`
     (`rde/model_adapters/models.py`) mirrors this concatenation exactly, in
     the same order, for both the declared feature width and the
     concatenated normalization statistics. A feature builder that only
     looked at the modality literally named `"robot_state"` would silently
     declare the wrong width the moment a second inline modality exists.
   - **`element_names` must be a flat, index-aligned list in the manifest.**
     Some source formats group per-dimension names by sub-limb (e.g.
     bimanual ALOHA's `{"motors": [...]}` rather than a flat list) —
     `rde.ingestion.lerobot._flatten_names` normalizes this at ingestion
     time; `rde.exploration.queries.dimensionality` tolerates the old,
     un-normalized shape defensively for manifests written before that fix.
     A dataset ingested before this normalization existed may still need
     that defensive path, not a re-ingestion, to profile correctly.
5. **Run the full training budget** once the dry run succeeds.

### 9.7 Run a data-fraction sweep

```python
from model_training.config import load_experiment_config
from model_training.experiments import run_data_fraction_sweep

base = load_experiment_config("configs/experiments/baseline.yaml")
results = run_data_fraction_sweep(base, fractions=[0.25, 0.5, 0.75, 1.0])
```

Each fraction runs as its own full experiment (own `experiment_id`, own run
directory); nothing here is a shortcut around `ExperimentRunner`.

## 10. Extending the System

### 10.1 Adding a new model

1. Implement `Model` (at minimum `predict`; `fit` defaults to a no-op) and,
   if it needs a tensor layout none of `StateAdapter`/`VisionStateAdapter`/
   `VisionLanguageAdapter` already produce, a matching `ModelInputAdapter`
   (Section 7.2) — reading shapes and keys from the dataset `contract`, never
   hard-coded modality names.
2. Register the pair in `MODEL_REGISTRY` (`rde.model_adapters.models`).
3. It is now usable everywhere a model name string is accepted:
   `evaluation.baseline_models` in a config, `evaluate_baselines`,
   `build_model`.

### 10.2 Adding an evaluation slice

Add a branch to `stratified_evaluate` (Section 6.6) resolving the slice name
against real per-episode metadata — the existing two (`per_task`,
`per_length_bucket`) are the pattern to follow. A slice requiring metadata
the dataset's schema does not carry must not be silently approximated; raise,
as the existing `else` branch does for an unknown name.

### 10.3 Cross-embodiment checklist

Before fine-tuning SmolVLA on a dataset with a different action
dimensionality, state width, or camera count than `so100_pickplace`:

- [ ] The dataset is ingested and its `clean/vN`/`training/vN` contract
      resolves without error via `mtrain profile`.
- [ ] `contract["action"]["shape"][0]` is ≤ the checkpoint's
      `max_action_dim` (32 in the current `lerobot/smolvla_base`).
- [ ] Every inline modality's normalization stats are present
      (`contract["observation"][k]["normalization"]`) — required for
      `_dataset_stats_from_contract` to build a fresh preprocessor.
- [ ] A short dry run (Section 9.6, step 3) has actually succeeded before a
      full-budget run is launched.

## 11. Related Documentation

**Table 21.** Where each adjacent question is answered.

| Document | Answers |
|---|---|
| `ARCHITECTURE.md` | Why the system is split this way; the full design rationale behind the `Dataset Version → DatasetView → ModelInputAdapter → Model → Prediction → Evaluation → Feedback` pipeline. |
| `EXPERIMENTS.md` | What the four intervention kinds (full/reduced/imbalanced/rebalanced) are for, and the quantity-vs-coverage study design. |
| `METRICS.md` | Definitions and statistical meaning of every metric `Evaluator` and this package compute. |
| `DEPLOYMENT.md` | The deployment runtime's design rationale, health checks, and latency measurement in depth. |
| `FAILURE_ANALYSIS.md` | The failure-extraction and ranking methodology behind `build_feedback_artifact` in depth. |
| `IMPLEMENTATION_PLAN.md` | Findings from the initial inspection of the data engine, the modifications made to it, and the resulting implementation plan. |
| `episodic-data-engine/docs/` | Ingestion, storage, QC, and versioning — everything left of `DatasetView` in the pipeline above. |
