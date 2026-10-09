# Deployment

## 1. Introduction

This document describes the deployment architecture implemented by
`model_training.deployment`. In accordance with the project specification's
instruction to avoid unnecessary infrastructure, the deployment path is a
single sequence of function calls—saved artifact, deterministic loader,
preprocessing, SmolVLA, action, metrics and logging—rather than a network
service, message queue, or orchestration layer.

## 2. Artifact Format

A checkpoint produced by `SmolVLAPolicy.save_checkpoint`
(`episodic-data-engine/rde/model_adapters/models.py`) is a directory
containing the five files listed in Table 1.

**Table 1.** Contents of a saved SmolVLA checkpoint directory.

| File | Content |
|---|---|
| `config.json` | The SmolVLA architecture configuration. |
| `model.safetensors` | The model weights. |
| `policy_preprocessor.json` | Normalization statistics and the language-tokenizer configuration. |
| `policy_postprocessor.json` | Action de-normalization parameters. |
| `rde_checkpoint_meta.json` | Checkpoint source, device, action dimensionality, chunk size, and any additional metadata supplied at save time (experiment identifier, step, dataset identity). |

The preprocessing and postprocessing pipelines are saved and reloaded
alongside the weights deliberately, because they carry the normalization
statistics and language tokenizer the checkpoint was trained with. Reloading
weights against a *different* pipeline—for example, the base model's
defaults—would silently misnormalize every input. `save_checkpoint`
therefore calls `self._pre.save_pretrained(path)` and
`self._post.save_pretrained(path)` in addition to saving the model weights.

`SmolVLAPolicy.verify_checkpoint(path)` checks that all five files are
present, together with at least one weights file, before any loading is
attempted. It raises `FileNotFoundError` or `ValueError`, naming the specific
missing file, on a corrupted, incomplete, or unrelated directory; no partial
load is ever attempted.

`Trainer.train` calls `verify_checkpoint` immediately after every
`save_checkpoint`, not only at load time: a save that failed partway
(disk full, OOM-killer, an interrupted process) is recorded with
`"verified": false` in `training_outcome.json` and logged to
`metrics/checkpoint_errors.jsonl`, and only a verified checkpoint is
eligible to become the run's `best` (by validation metric) — an unverified
save can neither be selected nor protected from pruning by virtue of being
`best`.

## 3. Model Loading Procedure

```python
from model_training.deployment import DeployedModel

model = DeployedModel(checkpoint_path, contract=view.contract).load()
prediction, latency = model.predict(samples)
```

`load()` invokes `health_check()` (Section 4) before constructing the model
via `SmolVLAPolicy.from_checkpoint`, which itself calls
`verify_checkpoint`. A checkpoint that fails verification never reaches
`predict()`.

## 4. Health Checks

Before any inference is attempted, `DeployedModel.health_check()` verifies
the following conditions, in order.

1. The checkpoint directory is complete, per `verify_checkpoint`.
2. The dataset contract declares a valid action shape.
3. The contract declares at least one image (external-plane) modality, with
   a valid three-dimensional shape.
4. Every inline (state) modality in the contract carries recorded
   normalization statistics.

Any failure raises `DeploymentHealthError`, or the more specific
`ValueError`/`FileNotFoundError` from checkpoint verification, with a
message naming the specific condition that failed. There is no fallback to a
different model: the project specification prohibits silently substituting
one model for another.

## 5. Preprocessing, Inference, and Postprocessing

`VisionLanguageAdapter.adapt`, reused without modification, performs image
decoding, resizing, tensor layout construction, and language-instruction
lookup, reading these from the dataset's training contract rather than a
hard-coded set of modality names; a different embodiment's contract is
therefore served by the same loader without code change.
`SmolVLAPolicy.predict` (via the `_map_inputs` helper shared with `fit`) maps
canonical camera names onto the checkpoint's positional input slots, applies
the checkpoint's own preprocessor (normalization and tokenization), invokes
`select_action`, and applies the postprocessor.

## 6. Latency Measurement

`DeployedModel.predict` returns a `LatencyBreakdown` object with the fields
defined in Table 2 of `docs/METRICS.md` (`model_load_ms`,
`preprocessing_ms`, `inference_ms`, `postprocessing_ms`, and `total_ms`).
Aggregate statistics across repeated calls—throughput, and the 50th, 95th,
and 99th percentile latencies—are left to the caller, since the number of
warm-up and measurement calls constitutes a benchmark-design decision this
project does not presume on the caller's behalf.

## 7. Model Registry

`model_training.deployment.ModelRegistry` is a lightweight local registry
class, implemented over the same append-only JSON Lines mechanism as the
rest of the tracking index, targeting `experiments/models.jsonl`. Table 2
lists its supported queries.

**Table 2.** Model registry queries.

| Question | Method |
|---|---|
| Which model is this? | `get(model_id)` |
| Which dataset trained it? | `row["dataset_id"]`, `row["dataset_version"]` |
| Which experiment created it? | `row["experiment_id"]` |
| Which checkpoint is deployable? | `row["artifact_path"]` |
| What metrics justified selection? | `row["metrics"]` |
| Which code version produced it? | `row["git_commit"]` |

`ModelRegistry` is not currently the path by which a model is actually
looked up. `ExperimentRunner.run` writes each model record directly through
`ExperimentIndex.record_model` (the same `experiments/models.jsonl` file,
via `JsonlIndex.append`) and never calls `ModelRegistry.register`. A caller
that queries `experiments/models.jsonl` directly, or via
`ExperimentIndex.models`, sees every registered model. A `ModelRegistry`
constructed on that same file returns the most recent row for a given
`model_id` from `get`, and is therefore consistent with the runner's records;
it is nonetheless not the wired-up registration path, because
`ExperimentRunner` never writes through it.

## 8. Scope Limitations

No network-facing server, message queue, or orchestration layer is
implemented. Should a network-facing interface become necessary in future
work, it would take the form of a thin wrapper around
`DeployedModel.predict`, not a rewrite of this layer.
