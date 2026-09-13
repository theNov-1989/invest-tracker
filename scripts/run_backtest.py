#!/usr/bin/env python
"""Thin CLI wrapper around the backtest runner.

Usage:
    python scripts/run_backtest.py --config configs/momentum_sp500.yaml --stage test

Prefer this (or ``python -m invest_tracker``) over importing internals directly.
Adds ``src/`` to the path so it runs without an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from invest_tracker.runner import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
