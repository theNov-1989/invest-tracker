"""Orchestration: wire the modules together into a single backtest run.

This is the one place the pieces meet. It stays deliberately thin -- load spec,
get data behind the interface, run the engine over a walk-forward stage, guard
against leakage, report -- so the interesting logic lives in the focused modules
it calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from invest_tracker.benchmark import buy_and_hold_returns
from invest_tracker.config import Spec, load_spec
from invest_tracker.data.base import PriceDataSource
from invest_tracker.data.yfinance_source import YFinanceSource
from invest_tracker.portfolio import BacktestResult, run_backtest
from invest_tracker.report import ComparisonReport, build_report, format_report
from invest_tracker.universe import load_universe
from invest_tracker.walkforward import Stage, WalkForwardSplit, split_from_spec


@dataclass(frozen=True)
class StageRun:
    stage: Stage
    result: BacktestResult
    benchmark_returns: pd.Series
    report: ComparisonReport


def run_stage(
    spec: Spec,
    split: WalkForwardSplit,
    prices: pd.DataFrame,
    stage: Stage,
) -> StageRun:
    """Run the strategy and benchmark for one walk-forward stage and report."""
    eval_start, eval_end = split.evaluation_window(stage)
    eval_start, eval_end = pd.Timestamp(eval_start), pd.Timestamp(eval_end)

    result = run_backtest(prices, spec, eval_start, eval_end)
    bench_returns = buy_and_hold_returns(prices, spec.benchmark, eval_start, eval_end)

    # Leakage tripwires: performance must be measured strictly within this stage.
    split.assert_within(result.net_returns, stage)
    split.assert_within(bench_returns, stage)

    report = build_report(
        strategy_returns=result.net_returns,
        benchmark_returns=bench_returns,
        stage=stage,
        benchmark_ticker=spec.benchmark,
        risk_free_rate=spec.metrics.risk_free_rate,
        periods_per_year=spec.metrics.periods_per_year,
    )
    return StageRun(stage=stage, result=result, benchmark_returns=bench_returns, report=report)


def run_from_config(
    config_path: str,
    stage: Stage = Stage.TEST,
    data_source: PriceDataSource | None = None,
) -> StageRun:
    """Full run from a config path: load, fetch, backtest one stage, report.

    Prices are fetched for the stage's ``price_window`` (evaluation window plus a
    warm-up buffer) so the signal has full lookback history on day one, while the
    reported window is only the stage's evaluation window.
    """
    spec = load_spec(config_path)
    split = split_from_spec(spec)
    source = data_source or YFinanceSource()

    tickers = load_universe(spec.universe)
    if spec.benchmark not in tickers:
        tickers = [*tickers, spec.benchmark]

    fetch_start, fetch_end = split.price_window(stage)
    prices = source.get_prices(tickers, fetch_start, fetch_end)

    return run_stage(spec, split, prices, stage)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="run-backtest",
        description="Run the walk-forward momentum backtest from a YAML spec.",
    )
    parser.add_argument(
        "--config",
        default="configs/momentum_sp500.yaml",
        help="Path to the YAML spec (default: configs/momentum_sp500.yaml).",
    )
    parser.add_argument(
        "--stage",
        choices=[s.value for s in Stage] + ["both"],
        default=Stage.TEST.value,
        help=(
            "Which walk-forward window to run. 'test' (default) is the honest "
            "out-of-sample result; 'tuning' is for parameter selection only; "
            "'both' runs each and reports them separately."
        ),
    )
    args = parser.parse_args(argv)

    stages = [Stage.TUNING, Stage.TEST] if args.stage == "both" else [Stage(args.stage)]

    for i, stage in enumerate(stages):
        run = run_from_config(args.config, stage=stage)
        if i > 0:
            print("\n" + "=" * 42 + "\n")
        print(format_report(run.report))

    return 0
