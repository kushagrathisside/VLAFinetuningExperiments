#!/usr/bin/env python
"""Thin wrapper: `python scripts/train.py --config configs/experiments/baseline.yaml`.
Identical to `mtrain train`; provided for environments where the console script is not on the path."""
import sys

from model_training.cli import app

if __name__ == "__main__":
    app(["train", *sys.argv[1:]])
