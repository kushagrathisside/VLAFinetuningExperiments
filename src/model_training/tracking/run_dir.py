"""Run directory creation and reproducibility metadata.

``runs/<date>/<experiment_name>/seed_<seed>/`` -- created once, never silently
reused: a run directory that already has content in it means a previous run is
there, and overwriting it would destroy the metrics/checkpoints/logs it holds.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _git_info(repo_path: Path) -> dict[str, Any]:
    if not (repo_path / ".git").exists():
        return {"repo": str(repo_path), "commit": None, "dirty": None, "branch": None,
                 "note": "not a git repository"}
    try:
        def _run(*args: str) -> str:
            return subprocess.check_output(
                ["git", "-C", str(repo_path), *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        commit = _run("rev-parse", "HEAD")
        dirty = bool(_run("status", "--porcelain"))
        branch = _run("rev-parse", "--abbrev-ref", "HEAD")
        return {"repo": str(repo_path), "commit": commit, "dirty": dirty, "branch": branch}
    except Exception as exc:  # pragma: no cover - environment-dependent
        return {"repo": str(repo_path), "commit": None, "dirty": None, "branch": None, "error": str(exc)}


def collect_git_info(rde_repo: Path, model_training_repo: Path) -> dict[str, Any]:
    """Git identity of both repos a run's code spans -- the engine and this layer."""
    return {
        "episodic_data_engine": _git_info(rde_repo),
        "vla_finetune": _git_info(model_training_repo),
    }


def _package_version(name: str) -> str | None:
    try:
        import importlib.metadata as im

        return im.version(name)
    except Exception:
        return None


def collect_environment() -> dict[str, Any]:
    """Every reproducibility field this process can actually observe.

    Fields that require a dependency not installed in the current environment
    (torch/lerobot in the light ``.venv``) are ``None`` with the reason stated,
    never fabricated.
    """
    try:
        import psutil
        ram_total_gb = round(psutil.virtual_memory().total / 1e9, 2)
    except ImportError:  # pragma: no cover
        ram_total_gb = None

    info: dict[str, Any] = {
        "python_version": sys.version,
        "os": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": ram_total_gb,
        "packages": {
            "episodic-data-engine": _package_version("episodic-data-engine"),
            "vla-finetune": _package_version("vla-finetune"),
            "numpy": _package_version("numpy"),
            "pydantic": _package_version("pydantic"),
        },
        "torch": None,
        "cuda": None,
        "gpu": None,
        "lerobot": None,
        "notes": [],
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = {"available": torch.cuda.is_available(), "version": torch.version.cuda}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "total_memory_gb": round(props.total_memory / 1e9, 2),
                "driver_capability": f"{props.major}.{props.minor}",
            }
    except ImportError:
        info["notes"].append("torch not installed in this environment (light .venv, no `vla` extra)")
    try:
        import lerobot

        info["lerobot"] = getattr(lerobot, "__version__", "installed, version unknown")
    except ImportError:
        info["notes"].append("lerobot not installed in this environment")
    return info


@dataclass
class RunDir:
    """One experiment run's artifact directory, in the standard run layout."""

    root: Path
    experiment_name: str
    seed: int
    date: str

    _SUBDIRS = ("checkpoints", "predictions", "evaluation", "failures", "reports", "logs", "metrics")

    @property
    def path(self) -> Path:
        return self.root / self.date / self.experiment_name / f"seed_{self.seed}"

    def ensure(self) -> Path:
        p = self.path
        p.mkdir(parents=True, exist_ok=True)
        for sub in self._SUBDIRS:
            (p / sub).mkdir(parents=True, exist_ok=True)
        return p

    def write_json(self, name: str, data: Any) -> Path:
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2, default=str))
        return target

    def append_jsonl(self, relpath: str, record: dict[str, Any]) -> None:
        target = self.path / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def sub(self, name: str) -> Path:
        p = self.path / name
        p.mkdir(parents=True, exist_ok=True)
        return p


def make_run_dir(
    base_runs_dir: Path, experiment_name: str, seed: int, date: str | None = None
) -> RunDir:
    """Create a fresh run directory. Raises if a non-empty one already exists.

    An existing experiment run is never silently overwritten. A
    same-day rerun of the same experiment+seed must be given a different
    ``experiment_name`` (e.g. a suffix) or explicitly clear the old directory
    first -- this function will not do either silently.
    """
    date = date or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    rd = RunDir(root=Path(base_runs_dir), experiment_name=experiment_name, seed=seed, date=date)
    if rd.path.exists() and any(rd.path.iterdir()):
        raise FileExistsError(
            f"run directory already exists and is non-empty: {rd.path} -- "
            "never silently overwrite an experiment run"
        )
    rd.ensure()
    return rd
