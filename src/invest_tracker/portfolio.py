"""Backtest / portfolio engine.

Monthly-rebalanced, long-only, equal-weight top-decile momentum portfolio.

On each rebalance date (the last trading day of each month within the evaluation
window) the engine:
  1. reads the momentum signal for that date,
  2. ranks the cross-section and selects the top ``long_quantile``,
  3. equal-weights the selected names,
  4. charges a turnover-based cost for moving from the old book to the new one,
  5. holds those weights until the next rebalance.

Daily portfolio return uses the weights held at the *previous* close applied to
today's return, so there is no look-ahead. Returns are reported net of costs.

Simplification (documented, not hidden): weights are assumed reset exactly to
target at each rebalance rather than tracking intra-month drift. For a monthly,
equal-weight, liquid large-cap book this slightly overstates turnover (hence
costs) rather than understating it -- a conservative direction for v1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from invest_tracker.config import Spec
from invest_tracker.costs import rebalance_cost
from invest_tracker.signal import compute_momentum


@dataclass(frozen=True)
class BacktestResult:
    """Output of a single backtest run over one evaluation window."""

    net_returns: pd.Series      # daily returns, net of costs
    gross_returns: pd.Series    # daily returns, before costs
    equity_curve: pd.Series     # cumulative net equity, starts at 1.0
    weights: pd.DataFrame       # target weights per rebalance date
    rebalance_dates: pd.DatetimeIndex


def _rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar month present in ``index``."""
    s = pd.Series(index, index=index)
    last = s.groupby(index.to_period("M")).last()
    return pd.DatetimeIndex(last.values)


def _target_weights(signal_row: pd.Series, long_quantile: float, columns: pd.Index) -> pd.Series:
    """Equal-weight the top ``long_quantile`` of a single-date signal cross-section."""
    valid = signal_row.dropna()
    weights = pd.Series(0.0, index=columns)
    if valid.empty:
        return weights
    n_long = max(1, int(np.floor(len(valid) * long_quantile)))
    winners = valid.nlargest(n_long).index
    weights.loc[winners] = 1.0 / n_long
    return weights


def run_backtest(
    prices: pd.DataFrame,
    spec: Spec,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
) -> BacktestResult:
    """Run the momentum backtest over ``[eval_start, eval_end]``.

    ``prices`` should include warm-up history *before* ``eval_start`` so the
    signal has full lookback on the first evaluation day (see
    :meth:`WalkForwardSplit.price_window`).
    """
    eval_start = pd.Timestamp(eval_start)
    eval_end = pd.Timestamp(eval_end)

    signal = compute_momentum(prices, spec.signal.lookback_days, spec.signal.skip_days)
    daily_returns = prices.pct_change()

    eval_days = prices.loc[eval_start:eval_end].index
    if len(eval_days) == 0:
        raise ValueError(f"No trading days in evaluation window [{eval_start}, {eval_end}].")

    rebal_dates = _rebalance_dates(eval_days)

    # Target weights per rebalance date, and the cost charged that day.
    target_weights = pd.DataFrame(0.0, index=rebal_dates, columns=prices.columns)
    cost_on_day = pd.Series(0.0, index=eval_days)
    prev_weights = pd.Series(0.0, index=prices.columns)

    for d in rebal_dates:
        new_weights = _target_weights(signal.loc[d], spec.portfolio.long_quantile, prices.columns)
        target_weights.loc[d] = new_weights.values
        cost_on_day.loc[d] = rebalance_cost(prev_weights, new_weights, spec.costs.per_trade_bps)
        prev_weights = new_weights

    # Held weights: forward-fill each rebalance's target across the days it is held.
    held = target_weights.reindex(eval_days).ffill().fillna(0.0)

    # Portfolio gross return on day t = weights held at t-1 applied to t's returns.
    aligned_returns = daily_returns.reindex(eval_days).fillna(0.0)
    gross = (held.shift(1).fillna(0.0) * aligned_returns).sum(axis=1)
    net = gross - cost_on_day

    equity = (1.0 + net).cumprod()

    return BacktestResult(
        net_returns=net,
        gross_returns=gross,
        equity_curve=equity,
        weights=target_weights,
        rebalance_dates=rebal_dates,
    )
