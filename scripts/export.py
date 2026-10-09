#!/usr/bin/env python
"""`python scripts/export.py --run runs/... --output paper_ready/` -- same as `mtrain export`."""
import sys

from model_training.cli import app

if __name__ == "__main__":
    app(["export", *sys.argv[1:]])
