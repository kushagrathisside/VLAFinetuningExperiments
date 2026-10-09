"""Independent re-verification of episode-level train/val/test disjointness.

``rde.dataset_builder.DatasetBuilder`` already guarantees this by construction
(``assign_splits`` partitions the episode set once, deterministically). This
check exists to catch a *real* leak regardless of
where it would come from -- including a bug this project's own intervention
builder introduces -- by re-deriving the partition independently from the
materialised :class:`~rde.engine.DatasetView` rather than trusting the code
path that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

from rde.engine import DatasetView


class SplitLeakageError(Exception):
    """Raised when any two of train/val/test share at least one episode id."""


@dataclass
class SplitIntegrityReport:
    dataset_id: str
    version: str
    split_sizes: dict[str, int]
    train_val_overlap: list[int]
    train_test_overlap: list[int]
    val_test_overlap: list[int]
    unassigned_episodes: list[int]

    @property
    def ok(self) -> bool:
        return not (self.train_val_overlap or self.train_test_overlap or self.val_test_overlap)

    def to_dict(self) -> dict:
        return {**self.__dict__, "ok": self.ok}


def verify_split_integrity(view: DatasetView, *, raise_on_failure: bool = True) -> SplitIntegrityReport:
    """Re-check train ∩ val = train ∩ test = val ∩ test = ∅ from scratch.

    Raises :class:`SplitLeakageError` on any overlap when ``raise_on_failure``
    (the default) -- fail loudly, never silently proceed with a
    contaminated split.
    """
    train = set(view.episode_ids(split="train"))
    val = set(view.episode_ids(split="val"))
    test = set(view.episode_ids(split="test"))
    all_ids = {e.episode_id for e in view.episodes}

    report = SplitIntegrityReport(
        dataset_id=view.dataset_id,
        version=view.version,
        split_sizes={"train": len(train), "val": len(val), "test": len(test)},
        train_val_overlap=sorted(train & val),
        train_test_overlap=sorted(train & test),
        val_test_overlap=sorted(val & test),
        unassigned_episodes=sorted(all_ids - train - val - test),
    )
    if raise_on_failure and not report.ok:
        raise SplitLeakageError(
            f"split leakage detected in {view.dataset_id}@{view.version}: "
            f"train∩val={report.train_val_overlap} train∩test={report.train_test_overlap} "
            f"val∩test={report.val_test_overlap}"
        )
    return report
