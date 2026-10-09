#!/usr/bin/env python
"""`python scripts/compare.py --runs a b c --metric mse` -- same as `mtrain compare`."""
import sys

from model_training.cli import app

if __name__ == "__main__":
    app(["compare", *sys.argv[1:]])
