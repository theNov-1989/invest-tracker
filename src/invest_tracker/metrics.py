"""Performance metrics.

Pure functions over a daily returns series (or its equity curve). All are used
identically for the strategy and the benchmark so they are compared on exactly
the same basis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceMetrics:
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def equity_curve(returns: pd.Series) -> pd.Series:
    """Cumulative equity from daily returns, starting at 1.0."""
    return (1.0 + returns.fillna(0.0)).cumprod()


def total_return(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float(equity_curve(returns).iloc[-1] - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty:
        return 0.0
    n = len(returns)
    growth = 1.0 + total_return(returns)
    if growth <= 0:
        return -1.0
    years = n / periods_per_year
    if years <= 0:
        return 0.0
    return float(growth ** (1.0 / years) - 1.0)


def sharpe(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio.

    ``risk_free_rate`` is an annualized decimal (0.0 = 0%); it is converted to a
    per-period rate before subtracting from returns.
    """
    if returns.empty:
        return 0.0
    per_period_rf = risk_free_rate / periods_per_year
    excess = returns.fillna(0.0) - per_period_rf
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Most negative peak-to-trough decline of the equity curve (<= 0)."""
    if returns.empty:
        return 0.0
    curve = equity_curve(returns)
    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0
    return float(drawdown.min())


def summarize(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> PerformanceMetrics:
    """Compute the full metrics bundle for a returns series."""
    return PerformanceMetrics(
        total_return=total_return(returns),
        cagr=cagr(returns, periods_per_year),
        sharpe=sharpe(returns, risk_free_rate, periods_per_year),
        max_drawdown=max_drawdown(returns),
    )
