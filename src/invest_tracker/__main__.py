"""Enable ``python -m invest_tracker`` as a backtest entry point."""

from __future__ import annotations

import sys

from invest_tracker.runner import main

if __name__ == "__main__":
    sys.exit(main())
