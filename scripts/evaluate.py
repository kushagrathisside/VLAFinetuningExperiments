#!/usr/bin/env python
"""`python scripts/evaluate.py --run runs/... --split test` -- same as `mtrain evaluate`."""
import sys

from model_training.cli import app

if __name__ == "__main__":
    app(["evaluate", *sys.argv[1:]])
