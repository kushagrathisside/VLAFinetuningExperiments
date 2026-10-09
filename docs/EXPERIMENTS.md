# Experiments

## 1. Introduction

This document describes the experiment configurations shipped with the
project, the rationale for the variable chosen in the imbalance and
rebalancing experiments, the mechanism supporting data-fraction studies, and
the policies governing evaluation slices, random seeds, and cross-experiment
comparison.

## 2. Primary Experiment Configurations

Four experiment configurations, corresponding to the project's primary
research comparison, are provided in `configs/experiments/`. Each is loaded
and validated by `model_training.config.load_experiment_config`, which
merges it onto `configs/base.yaml` before validation. Table 1 summarizes
their status.

**Table 1.** Primary experiment configurations.

| Configuration | Intervention | Status |
|---|---|---|
| `baseline.yaml` | `full` (every valid demonstration) | Runnable as provided. |
| `reduced.yaml` | `reduced` (a seeded random draw of fifty percent of episodes) | Runnable as provided. |
| `imbalanced.yaml` | `custom`, with an empty episode list | A template. Must be resolved with `mtrain build-intervention` against an actual dataset profile before use. |
| `rebalanced.yaml` | `rebalanced`, with empty episode weights | A template that deliberately fails to *load*, not merely to resolve, until its intervention block is filled in. |

`baseline.yaml` and `reduced.yaml` require no knowledge of the dataset's
content: "every episode" and "a seeded random half" are both well-defined
without inspecting the data. `imbalanced.yaml` and `rebalanced.yaml` are
templates because the project specification requires that the dimension
along which they are constructed be determined from an actual profile of the
dataset, rather than presumed in advance.

## 3. Selection of the Imbalance Variable

The dataset's task identity, the most obvious candidate for a "variation"
axis, does not exist in a usable form for the project's primary dataset: a
profile of `so100_pickplace` (via `mtrain profile --dataset so100_pickplace
--version training/v1`) reports a single distinct task instruction across
all forty-eight episodes. Task-identity-based imbalance is implemented and
remains available for datasets that vary by task (`aloha_coffee`, for
instance), but has nothing to select between on the primary dataset.

Trajectory length is the variable used in practice. The profile confirms a
genuine, non-degenerate spread—326 to 475 frames per episode, with a mean
duration of approximately thirteen seconds. `episode_length_bucket`
(`model_training.data.interventions.compute_length_bucket`) partitions
episodes into short, medium, and long groups by within-dataset terciles and
is the variable the imbalanced and rebalanced configurations resolve to. The
resolution is performed explicitly, not embedded in the configuration
loader:

```bash
mtrain build-intervention \
  --dataset so100_pickplace --version clean/v1 \
  --kind imbalanced --variable episode_length_bucket --variable-value short \
  --target-config configs/experiments/imbalanced.yaml --apply
```

This procedure is documented here, rather than hard-coded, because the
project specification requires the choice to follow from an actual profile
of the actual data. Selecting the variable once, by hand, from the printed
profile, and recording the resulting episode identifiers together with a
stated reason, constitutes the auditable form of that requirement.

## 4. Constraints on the Rebalancing Intervention

A rebalanced dataset variant can only upweight real, existing episodes that
share the targeted attribute; the mechanism is the per-episode
`sampling_weight` already supported by `rde.dataset_builder.DatasetBuilder
.build`, consumed at training time by `SmolVLAPolicy.fit`'s weighted sampler.
Because this project does not collect new demonstrations, a rebalanced run
cannot close an actual coverage gap—it can only redistribute training
attention over data that already exists. Every experiment report generated
by the system states this limitation explicitly in its "Limitations"
section. A rebalanced result should not be interpreted as evidence that a
coverage gap was closed.

## 5. Data-Fraction Experiments

`model_training.experiments.run_data_fraction_sweep` executes one experiment
per requested data fraction, holding every other parameter—model,
hyperparameters, splits, and seed—fixed, and varying only the intervention
(`{"kind": "fraction", "fraction": f}`). No part of the training code differs
between fractions; only the dataset intervention does. The function does not
run automatically: the caller supplies the fraction list and accepts the
resulting computational cost.

## 6. Fixed Evaluation Slices

Every experiment run evaluates the test split of its own `training/vN`
version (assigned once, by episode, via a seeded hash), together with
whichever named slices `evaluation.slices` requests. Two slices are
supported.

**Table 2.** Supported evaluation slices.

| Slice | Definition |
|---|---|
| `per_task` | One slice per distinct task identifier present in the split. |
| `per_length_bucket` | Short, medium, and long groups by within-split episode-length terciles. |

A slice is a frozen episode-identifier list, produced identically on every
invocation of `model_training.evaluation.offline.stratified_evaluate`.
Re-evaluating the same `training/vN` version always scores the same
episodes. Comparing two different dataset interventions is, necessarily, a
comparison across two different `training/vN` versions' test splits, since
an intervention can change which episodes are assigned to `test` in the
first place; each experiment's `dataset.json` records exactly which version
and which episode identifiers were evaluated, so this distinction is always
recoverable.

## 7. Seed Policy

`ExperimentConfig.seeds` is a list. `ExperimentRunner.run(config, seed=...)`
selects one seed per invocation, defaulting to the first entry of the list.
Each invocation produces its own
`runs/<date>/<experiment_name>/seed_<seed>/` directory and never overwrites a
sibling directory for a different seed. Running the same configuration
across several seeds is a matter of calling `run` once per seed; `mtrain
compare` can then report the mean and standard deviation across the
resulting run directories. Where only a single seed has been executed, the
comparison states a single-seed result rather than a confidence interval
fabricated from one observation.

## 8. Cross-Experiment Comparison

```bash
mtrain compare --runs runs/.../baseline_full/seed_0 --runs runs/.../reduced_data/seed_0 --metric mse
```

reads each run's saved `evaluation/test_overall.json` and
`evaluation/baselines.json`—never live process state—and renders the
comparison table described in the project specification. The `--out` flag
additionally writes the same rows to a comma-separated file.
