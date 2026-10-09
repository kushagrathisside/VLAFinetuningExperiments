"""Experiment configuration schema.

Every experimental parameter that matters lives here, not in Python source.
An :class:`ExperimentConfig` is the single object a run is reproducible
from: dataset identity + intervention, splits, training hyperparameters,
evaluation configuration, seeds and tags. It is loaded from YAML and the fully
*resolved* object (defaults included) is what gets persisted alongside a run --
see :mod:`model_training.tracking.run_dir`.

Deliberately not config for its own sake: nothing here duplicates a concept the
Data Engine already owns. ``dataset`` names a *version*, never a path (mirrors
``rde.engine.DataEngine.get_dataset``); ``splits`` names the same three knobs
``rde.dataset_builder.DatasetBuilder.build`` accepts, because a training run's
splits and the dataset version's splits are the same thing viewed from two call
sites, not two schemas that could silently disagree.
"""

from __future__ import annotations

import hashlib
import datetime as _dt
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class DatasetRef(BaseModel):
    """The dataset version an experiment starts from -- an identity, not a path."""

    dataset_id: str
    version: str = Field(..., description="e.g. 'clean/v1' -- the input to the intervention/build step")


class DatasetIntervention(BaseModel):
    """What, if anything, changes about the data before training.

    ``kind`` names one of this project's four primary experiments plus
    the general ``fraction`` mechanism data-efficiency curves need:

    ``full``        every valid demonstration; no episode selection.
    ``fraction``     an explicit fraction of episodes (data-efficiency curve).
    ``reduced``       shorthand for ``fraction`` at a project-standard ~50%.
    ``imbalanced``     a deliberately skewed subset along ``variable``.
    ``rebalanced``      upweights (never duplicates or fabricates) the episodes
                         belonging to ``variable``'s under-represented value(s),
                         via per-episode sampling weight -- see
                         ``rde.dataset_builder.DatasetBuilder.build``'s
                         ``episode_weights``.
    ``custom``            an explicit ``episode_ids`` list, for anything the
                           above don't name.

    ``variable`` is deliberately left unset by default: the design forbids
    presuming the weak dimension in advance. It must be filled from an actual
    dataset profile (``model_training.data.profile``) before an ``imbalanced``/
    ``rebalanced`` run resolves, or resolution fails loudly rather than guessing.
    """

    kind: Literal["full", "fraction", "reduced", "imbalanced", "rebalanced", "custom"] = "full"
    fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    variable: str | None = None
    variable_value: Any = None
    episode_ids: list[int] | None = None
    episode_weights: dict[int, float] | None = None
    oversample_factor: float = 3.0
    seed: int = 0
    reason: str = ""

    @model_validator(mode="after")
    def _check_shape(self) -> "DatasetIntervention":
        if self.kind in ("fraction", "reduced") and self.episode_ids is None and self.fraction is None:
            if self.kind == "reduced":
                self.fraction = 0.5
            else:
                raise ValueError("intervention.kind='fraction' requires 'fraction' or an explicit 'episode_ids'")
        if self.kind == "imbalanced" and self.episode_ids is None and self.variable is None:
            raise ValueError(
                "intervention.kind='imbalanced' requires 'variable' (from a dataset profile) "
                "or an explicit 'episode_ids' -- the weak dimension must not be presumed"
            )
        if self.kind == "rebalanced" and self.variable is None and not self.episode_weights:
            raise ValueError(
                "intervention.kind='rebalanced' requires 'variable' (from a dataset profile) "
                "or an explicit non-empty 'episode_weights' -- nothing to rebalance towards otherwise"
            )
        if self.kind == "custom" and self.episode_ids is None:
            raise ValueError("intervention.kind='custom' requires an explicit 'episode_ids'")
        return self


class SplitConfig(BaseModel):
    """Episode-level train/val/test fractions -- identical semantics to
    :func:`rde.transformations.ops.assign_splits`, because it *is* that call."""

    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 42
    mode: Literal["quantile", "independent"] = "quantile"

    @model_validator(mode="after")
    def _sums_to_one(self) -> "SplitConfig":
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train+val+test must sum to 1.0, got {total}")
        return self

    def as_fractions(self) -> dict[str, float]:
        return {"train": self.train, "val": self.val, "test": self.test}


class TrainingConfig(BaseModel):
    """Fine-tuning hyperparameters, passed almost verbatim to
    ``SmolVLAPolicy.fit`` (:mod:`rde.model_adapters.models`)."""

    model: str = "smolvla"
    pretrained_checkpoint: str = "lerobot/smolvla_base"
    resume_from: str | None = None

    steps: int = 200
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 0

    checkpoint_every: int = 50
    validate_every: int = 50
    val_batches: int = 8
    keep_last_n_checkpoints: int = 3

    num_workers: int = 0
    mixed_precision: bool = False
    device: str | None = None  # None => auto (cuda if available, else cpu)

    seed: int = 0


class EvaluationConfig(BaseModel):
    """Offline evaluation + model-selection configuration."""

    split: str = "test"
    batch_size: int = 32
    max_frames: int | None = None
    success_tolerance: float = 0.1
    baseline_models: list[str] = Field(default_factory=lambda: ["mean_action", "nearest_neighbor"])

    #: Metric used for checkpoint selection. Must be computed on validation,
    #: never test -- enforced by model_training.experiments.selection, not here.
    selection_metric: str = "mse"
    selection_mode: Literal["min", "max"] = "min"

    #: Names of persistent evaluation slices to additionally score at test time,
    #: beyond the plain overall split. Resolved against the dataset profile /
    #: fixed slice registry, e.g. ["per_task", "per_variation", "hard_cases"].
    slices: list[str] = Field(default_factory=lambda: ["per_task"])


class ExperimentConfig(BaseModel):
    """The object one experiment is reproducible from."""

    experiment_name: str
    hypothesis: str = ""
    objective: str = ""

    dataset: DatasetRef
    intervention: DatasetIntervention = Field(default_factory=DatasetIntervention)
    splits: SplitConfig = Field(default_factory=SplitConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    seeds: list[int] = Field(default_factory=lambda: [0])
    tags: dict[str, str] = Field(default_factory=dict)

    #: Set once at load time; not user-editable via YAML.
    experiment_id: str | None = None
    created_at: str | None = None
    source_config_path: str | None = None

    @model_validator(mode="after")
    def _stamp_identity(self) -> "ExperimentConfig":
        if self.created_at is None:
            self.created_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
        if self.experiment_id is None:
            basis = f"{self.experiment_name}:{self.dataset.dataset_id}:{self.dataset.version}:{self.created_at}"
            digest = hashlib.sha256(basis.encode()).hexdigest()[:10]
            self.experiment_id = f"{self.experiment_name}-{digest}"
        return self

    def resolved_dict(self) -> dict[str, Any]:
        """Every field, defaults included -- what gets persisted as
        ``config.resolved.yaml`` so a run never depends on an unwritten default."""
        return self.model_dump(mode="json")
