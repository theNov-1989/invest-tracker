#!/usr/bin/env python
"""Compact baseline-backtest summary.

Prints only:
  * the cost assumption (bps) and both walk-forward window date ranges,
  * OUT-OF-SAMPLE (net of costs) CAGR / Sharpe / max drawdown for the strategy
    and for SPY buy-and-hold over the identical window,
  * the strategy's TUNING-window CAGR (to read the tuning -> OOS decay),
  * a plain verdict: did the strategy beat SPY out-of-sample, net of costs.

Nothing else. Usage:
    python scripts/summary.py --config configs/momentum_sp500.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from invest_tracker.config import load_spec  # noqa: E402
from invest_tracker.data.base import PriceDataSource  # noqa: E402
from invest_tracker.data.yfinance_source import YFinanceSource  # noqa: E402
from invest_tracker.runner import run_stage  # noqa: E402
from invest_tracker.walkforward import Stage, split_from_spec  # noqa: E402


def build_summary(config_path: str, data_source: PriceDataSource | None = None) -> str:
    spec = load_spec(config_path)
    split = split_from_spec(spec)
    source = data_source or YFinanceSource()

    tickers = load_universe_tickers(spec)

    # Fetch once over the union of both windows' price ranges (incl. warm-up),
    # then evaluate each stage from the same panel.
    tune_fetch_start, _ = split.price_window(Stage.TUNING)
    prices = source.get_prices(tickers, tune_fetch_start, spec.period.end)

    test = run_stage(spec, split, prices, Stage.TEST)
    tuning = run_stage(spec, split, prices, Stage.TUNING)

    s = test.report.strategy
    b = test.report.benchmark
    tune_cagr = tuning.report.strategy.cagr

    tune_lo, tune_hi = split.evaluation_window(Stage.TUNING)
    test_lo, test_hi = split.evaluation_window(Stage.TEST)
    beat = test.report.beats_benchmark

    def pct(x: float) -> str:
        return f"{x * 100:.2f}"

    line = "=" * 60
    verdict = (
        "Strategy BEAT SPY out-of-sample, net of costs."
        if beat
        else "Strategy did NOT beat SPY out-of-sample, net of costs."
    )
    return "\n".join(
        [
            line,
            " Momentum backtest — compact summary",
            line,
            f" Cost assumption : {spec.costs.per_trade_bps:g} bps of turnover, "
            f"charged each rebalance",
            f" Tuning window   : {tune_lo} -> {tune_hi}",
            f" OOS test window : {test_lo} -> {test_hi}",
            "",
            " OUT-OF-SAMPLE (net of costs)",
            f"   {'':<12}{'CAGR %':>9}{'Sharpe':>9}{'MaxDD %':>10}",
            f"   {'Strategy':<12}{pct(s.cagr):>9}{s.sharpe:>9.2f}{pct(s.max_drawdown):>10}",
            f"   {'SPY (B&H)':<12}{pct(b.cagr):>9}{b.sharpe:>9.2f}{pct(b.max_drawdown):>10}",
            "",
            f" Tuning-window strategy CAGR : {pct(tune_cagr)} %"
            f"   (OOS CAGR {pct(s.cagr)} % -> tuning->OOS decay)",
            "",
            f" Verdict: {verdict}",
            line,
        ]
    )


def load_universe_tickers(spec) -> list[str]:
    from invest_tracker.universe import load_universe

    tickers = load_universe(spec.universe)
    if spec.benchmark not in tickers:
        tickers = [*tickers, spec.benchmark]
    return tickers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="summary",
        description="Print a compact OOS-vs-SPY backtest summary with tuning-window CAGR.",
    )
    parser.add_argument(
        "--config",
        default="configs/momentum_sp500.yaml",
        help="Path to the YAML spec (default: configs/momentum_sp500.yaml).",
    )
    args = parser.parse_args(argv)
    print(build_summary(args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
