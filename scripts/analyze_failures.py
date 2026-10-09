#!/usr/bin/env python
"""`python scripts/analyze_failures.py --run runs/...` -- same as `mtrain analyze-failures`."""
import sys

from model_training.cli import app

if __name__ == "__main__":
    app(["analyze-failures", *sys.argv[1:]])
