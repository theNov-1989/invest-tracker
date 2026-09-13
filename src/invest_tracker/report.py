"""Reporting: side-by-side strategy vs benchmark, and the explicit verdict.

The success criterion lives here in code, unambiguously:

    The strategy must beat SPY buy-and-hold, out-of-sample (the TEST window),
    net of costs -- measured by total return over the window.

``report.py`` prints a comparison table and a single PASS/FAIL line so a run's
conclusion cannot be read two ways.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from invest_tracker.metrics import PerformanceMetrics, summarize
from invest_tracker.walkforward import Stage


@dataclass(frozen=True)
class ComparisonReport:
    stage: Stage
    strategy: PerformanceMetrics
    benchmark: PerformanceMetrics
    benchmark_ticker: str
    beats_benchmark: bool   # strategy total return > benchmark total return

    def verdict_line(self) -> str:
        oos = " (OUT-OF-SAMPLE)" if self.stage is Stage.TEST else ""
        outcome = "PASS" if self.beats_benchmark else "FAIL"
        return (
            f"[{outcome}] Strategy {'beats' if self.beats_benchmark else 'does NOT beat'} "
            f"{self.benchmark_ticker} buy-and-hold net of costs on the "
            f"{self.stage.value} window{oos}."
        )


def build_report(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    stage: Stage,
    benchmark_ticker: str,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> ComparisonReport:
    strat = summarize(strategy_returns, risk_free_rate, periods_per_year)
    bench = summarize(benchmark_returns, risk_free_rate, periods_per_year)
    return ComparisonReport(
        stage=stage,
        strategy=strat,
        benchmark=bench,
        benchmark_ticker=benchmark_ticker,
        beats_benchmark=strat.total_return > bench.total_return,
    )


def _fmt_pct(x: float) -> str:
    return f"{x * 100:8.2f}%"


def format_report(report: ComparisonReport) -> str:
    """Render a human-readable comparison table plus the verdict line."""
    s, b = report.strategy, report.benchmark
    rows = [
        ("Total return", _fmt_pct(s.total_return), _fmt_pct(b.total_return)),
        ("CAGR", _fmt_pct(s.cagr), _fmt_pct(b.cagr)),
        ("Sharpe", f"{s.sharpe:9.2f}", f"{b.sharpe:9.2f}"),
        ("Max drawdown", _fmt_pct(s.max_drawdown), _fmt_pct(b.max_drawdown)),
    ]
    label = f"{report.benchmark_ticker} (B&H)"
    lines = [
        f"Walk-forward stage: {report.stage.value.upper()}",
        "",
        f"{'Metric':<16}{'Strategy':>12}{label:>14}",
        "-" * 42,
    ]
    lines += [f"{name:<16}{strat:>12}{bench:>14}" for name, strat, bench in rows]
    lines += ["", report.verdict_line()]
    return "\n".join(lines)
