"""Momentum signal computation.

The signal is 12-month price momentum skipping the most recent month: for each
ticker on each date ``t``, the return from ``t - lookback_days`` to
``t - skip_days`` (in trading days).

These are pure functions of a price panel -- no I/O, no config reads, no
look-ahead -- which makes them straightforward to unit-test with known answers.
The "skip" avoids the well-documented short-term reversal effect in the most
recent month.
"""

from __future__ import annotations

import pandas as pd


def compute_momentum(
    prices: pd.DataFrame,
    lookback_days: int,
    skip_days: int,
) -> pd.DataFrame:
    """Compute the momentum signal for every date and ticker.

    Args:
        prices: DataFrame of adjusted closes, indexed by date, columns = tickers.
        lookback_days: formation window length in trading days (e.g. 252).
        skip_days: trailing days to skip (e.g. 21).

    Returns:
        DataFrame aligned to ``prices`` where entry ``(t, ticker)`` is
        ``price[t - skip_days] / price[t - lookback_days] - 1``. The first
        ``lookback_days`` rows are NaN (insufficient history) and never traded.

    The signal at ``t`` uses only prices at or before ``t - skip_days``, so it is
    strictly backward-looking and safe to act on at the close of ``t``.
    """
    if lookback_days <= 0 or skip_days < 0:
        raise ValueError("lookback_days must be > 0 and skip_days >= 0.")
    if skip_days >= lookback_days:
        raise ValueError("skip_days must be smaller than lookback_days.")

    start_price = prices.shift(lookback_days)
    end_price = prices.shift(skip_days)
    return end_price / start_price - 1.0


def momentum_as_of(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_days: int,
    skip_days: int,
) -> pd.Series:
    """Convenience: the momentum signal for all tickers on a single date.

    Returns a Series indexed by ticker (NaNs dropped) for the row at ``as_of``.
    Raises KeyError if ``as_of`` is not a trading day present in ``prices``.
    """
    signal = compute_momentum(prices, lookback_days, skip_days)
    if as_of not in signal.index:
        raise KeyError(f"{as_of} is not a trading day in the price index.")
    return signal.loc[as_of].dropna()
