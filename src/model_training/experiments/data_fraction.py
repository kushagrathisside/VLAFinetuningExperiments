"""Data-fraction sweep driver.

Makes the fraction/count/coverage/validation-metric/test-metric table possible
without rewriting training code: each fraction is just a
``DatasetIntervention(kind="fraction", fraction=f)`` run through the same
:class:`~model_training.experiments.runner.ExperimentRunner`. Fractions are not
executed unless explicitly requested -- calling this function *does* run them
(the framework makes them possible; they do not run by default); the caller decides which fractions and how many.
"""

from __future__ import annotations

from typing import Any

from ..config.schema import ExperimentConfig
from .runner import ExperimentRunner


def run_data_fraction_sweep(
    base_config: ExperimentConfig,
    fractions: list[float],
    runner: ExperimentRunner | None = None,
) -> list[dict[str, Any]]:
    """Run ``base_config`` once per fraction, varying only the data quantity.

    Every other knob (model, hyperparameters, splits, seed) is held fixed, so
    the resulting curve isolates the quantity effect -- exactly the
    "quantity vs. coverage" question this project is designed to
    answer, not merely "more data vs. less data".
    """
    runner = runner or ExperimentRunner()
    results: list[dict[str, Any]] = []
    for frac in fractions:
        data = base_config.model_dump()
        data["experiment_name"] = f"{base_config.experiment_name}_frac{int(round(frac * 100))}"
        data["intervention"] = {"kind": "fraction", "fraction": frac, "seed": base_config.intervention.seed}
        data["experiment_id"] = None
        data["created_at"] = None
        cfg = ExperimentConfig.model_validate(data)
        outcome = runner.run(cfg)
        results.append({"fraction": frac, **outcome.summary})
    return results
