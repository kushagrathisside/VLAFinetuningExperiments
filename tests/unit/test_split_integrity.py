from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from rde.engine import DataEngine

from model_training.data import SplitLeakageError, verify_split_integrity


def test_clean_training_version_has_no_leakage(settings, catalog, training_version):
    view = DataEngine(settings).get_dataset("so100_test", training_version)
    report = verify_split_integrity(view)
    view.close()
    assert report.ok
    assert sum(report.split_sizes.values()) == 8


@dataclass
class _FakeEpisode:
    episode_id: int


@dataclass
class _FakeView:
    """Minimal stand-in exposing only what verify_split_integrity reads --
    used to prove the checker actually detects an engineered overlap, since a
    real DatasetView built by rde's own (correct) builder cannot produce one."""

    dataset_id: str
    version: str
    _splits: dict[str, list[int]]

    @property
    def episodes(self):
        ids = sorted({i for v in self._splits.values() for i in v})
        return [_FakeEpisode(i) for i in ids]

    def episode_ids(self, split=None):
        return list(self._splits.get(split, []))


def test_split_leakage_is_detected_and_raises():
    leaky = _FakeView(
        dataset_id="x", version="y",
        _splits={"train": [1, 2, 3], "val": [4, 5], "test": [3, 6]},  # 3 is in both train and test
    )
    with pytest.raises(SplitLeakageError, match=r"train.test"):
        verify_split_integrity(leaky)  # type: ignore[arg-type]


def test_split_leakage_report_can_be_inspected_without_raising():
    leaky = _FakeView(
        dataset_id="x", version="y",
        _splits={"train": [1, 2], "val": [2, 3], "test": [4]},  # 2 is in both train and val
    )
    report = verify_split_integrity(leaky, raise_on_failure=False)  # type: ignore[arg-type]
    assert not report.ok
    assert report.train_val_overlap == [2]
    assert report.train_test_overlap == []
    assert report.val_test_overlap == []


def test_clean_splits_report_ok(settings, catalog, training_version):
    view = DataEngine(settings).get_dataset("so100_test", training_version)
    report = verify_split_integrity(view, raise_on_failure=False)
    view.close()
    assert report.ok
    assert report.unassigned_episodes == []
