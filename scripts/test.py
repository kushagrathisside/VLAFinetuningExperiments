#!/usr/bin/env python
"""`python scripts/test.py --run runs/...` -- same as `mtrain test`."""
import sys

from model_training.cli import app

if __name__ == "__main__":
    app(["test", *sys.argv[1:]])
