"""Local deployment runtime: deterministic loader, health checks, and
per-stage latency measurement.

Deliberately just a function call -- ``saved artifact -> loader ->
preprocessing -> SmolVLA -> Prediction`` -- no web service, no queue, no
orchestration layer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from rde.engine import Sample
from rde.model_adapters.adapters import VisionLanguageAdapter
from rde.model_adapters.base import Prediction
from rde.model_adapters.models import SmolVLAPolicy


class DeploymentHealthError(Exception):
    """Raised by :meth:`DeployedModel.health_check` on any checkpoint/contract
    mismatch, detected before inference rather than surfacing as a confusing
    tensor-shape error mid-prediction."""


@dataclass
class LatencyBreakdown:
    model_load_ms: float | None
    preprocessing_ms: float
    inference_ms: float
    postprocessing_ms: float
    total_ms: float
    #: What "preprocessing_ms" actually covers -- image decode, resize and
    #: tensor construction are not separately instrumented inside
    #: rde.model_adapters.adapters.VisionStateAdapter, so they are reported as
    #: one bucket rather than fabricating a finer split that this runtime
    #: does not measure.
    note: str = "preprocessing_ms includes image decode + resize + tensor construction (not separately instrumented)"

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__}


class DeployedModel:
    """Loads a checkpoint once, verifies it against a live dataset contract,
    and serves predictions with latency measurement."""

    def __init__(self, checkpoint_path: str, contract: dict[str, Any], device: str | None = None):
        self.checkpoint_path = str(checkpoint_path)
        self.contract = contract
        self.device = device
        self._policy: SmolVLAPolicy | None = None
        self._adapter = VisionLanguageAdapter(contract)
        self.load_time_ms: float | None = None

    def health_check(self) -> dict[str, Any]:
        """Everything that must be verified before inference.

        Raises :class:`DeploymentHealthError` with a specific, actionable
        reason on the first mismatch found -- never silently proceeds and
        never falls back to a different model.
        """
        meta = SmolVLAPolicy.verify_checkpoint(self.checkpoint_path)  # raises on incomplete/foreign dir

        action = self.contract.get("action", {})
        if not action.get("shape"):
            raise DeploymentHealthError("dataset contract has no action shape -- cannot verify compatibility")

        obs = self.contract.get("observation", {})
        images = [k for k, v in obs.items() if v.get("plane") == "external"]
        if not images:
            raise DeploymentHealthError(
                "dataset contract has no image modality -- SmolVLA requires at least one camera"
            )
        for k in images:
            shape = obs[k].get("shape")
            if not shape or len(shape) != 3:
                raise DeploymentHealthError(f"camera {k!r} has no valid (H, W, C) shape in the contract: {shape}")

        state_keys = [k for k, v in obs.items() if v.get("plane") == "inline"]
        for k in state_keys:
            if not obs[k].get("normalization"):
                raise DeploymentHealthError(
                    f"state modality {k!r} carries no normalization statistics in the dataset contract -- "
                    "the checkpoint's preprocessor expects normalized state and per-run normalization is missing"
                )

        return {"status": "ok", "checkpoint_meta": meta, "cameras": images, "state_keys": state_keys}

    def load(self) -> "DeployedModel":
        self.health_check()
        t0 = time.perf_counter()
        self._policy = SmolVLAPolicy.from_checkpoint(self.checkpoint_path, device=self.device)
        self.load_time_ms = (time.perf_counter() - t0) * 1000.0
        return self

    def predict(self, samples: Sequence[Sample]) -> tuple[Prediction, LatencyBreakdown]:
        """Run one inference call, returning the prediction plus a latency breakdown."""
        if self._policy is None:
            raise RuntimeError("call load() before predict()")
        t0 = time.perf_counter()
        model_input = self._adapter.adapt(samples)
        t1 = time.perf_counter()
        prediction = self._policy.timed_predict(model_input)
        t2 = time.perf_counter()

        preprocessing_ms = (t1 - t0) * 1000.0
        inference_ms = prediction.latency_ms
        postprocessing_ms = max(0.0, (t2 - t1) * 1000.0 - inference_ms)
        latency = LatencyBreakdown(
            model_load_ms=self.load_time_ms,
            preprocessing_ms=preprocessing_ms,
            inference_ms=inference_ms,
            postprocessing_ms=postprocessing_ms,
            total_ms=preprocessing_ms + inference_ms + postprocessing_ms,
        )
        return prediction, latency
