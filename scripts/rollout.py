#!/usr/bin/env python
"""`python scripts/rollout.py --run runs/...` -- same as `mtrain rollout`.
Reports "unavailable": no simulator/robot is connected in this environment."""
import sys

from model_training.cli import app

if __name__ == "__main__":
    app(["rollout", *sys.argv[1:]])
