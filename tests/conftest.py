"""Shared fixtures: a real synthetic dataset taken through raw -> clean ->
training, exactly the way `rde run` would, so every layer of model_training is
tested against real rde objects rather than mocks."""

from __future__ import annotations

from pathlib import Path

import pytest
from rde.catalog import Catalog
from rde.config import Paths, Settings, set_settings
from rde.dataset_builder import DatasetBuilder
from rde.ingestion import ingest
from rde.qc import QCEngine
from rde.testing import make_lerobot_dataset
from rde.transformations import NormalizeVector, TransformPipeline


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(paths=Paths(tmp_path / "rde_data"), row_group_size=64, batch_size=32)
    s.paths.ensure()
    set_settings(s)
    return s


@pytest.fixture
def catalog(settings: Settings) -> Catalog:
    return Catalog(settings.paths.catalog_db)


@pytest.fixture
def source_dataset(tmp_path: Path) -> Path:
    # Two tasks (not the single-task default) so episode_id-by-task_id
    # selection -- what the imbalanced/rebalanced interventions use -- has
    # something real to select between.
    return make_lerobot_dataset(
        tmp_path / "source", num_episodes=8, frames_per_episode=20, fps=10,
        tasks=("Pick up the cube and place it in the box.", "Push the block to the target."),
    )


@pytest.fixture
def raw(settings: Settings, source_dataset: Path, catalog: Catalog):
    return ingest(source_dataset, "so100_test", "lerobot", "raw/v1", settings=settings, catalog=catalog)


@pytest.fixture
def clean_version(settings: Settings, catalog: Catalog, raw) -> str:
    TransformPipeline(
        [NormalizeVector("state.joint"), NormalizeVector("action.joint")],
        settings=settings, catalog=catalog,
    ).run("so100_test", "raw/v1", "clean/v1")
    # `rde run` always validates before a version is considered usable
    # downstream; without this, every episode sits at quality_status=UNKNOWN
    # and DatasetBuilder.build() (which admits only PASSED/WARNING by
    # default) would refuse to build a training version from it.
    QCEngine(settings=settings, catalog=catalog).run("so100_test", "clean/v1")
    return "clean/v1"


@pytest.fixture
def training_version(settings: Settings, catalog: Catalog, clean_version: str) -> str:
    DatasetBuilder(settings=settings, catalog=catalog).build(
        "so100_test", clean_version, "training/v1",
        splits={"train": 0.6, "val": 0.2, "test": 0.2}, seed=7,
    )
    return "training/v1"
