"""Benchmark: buy-and-hold of a single ticker (SPY by default).

Computed over the exact same evaluation window as the strategy so the two are
directly comparable, net-of-costs strategy vs a passive hold with no rebalancing.
"""

from __future__ import annotations

import pandas as pd


def buy_and_hold_returns(
    prices: pd.DataFrame,
    ticker: str,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
) -> pd.Series:
    """Daily returns of a buy-and-hold position in ``ticker`` over the window.

    A single buy at the start incurs no ongoing turnover, so no cost model is
    applied -- this is the honest passive baseline the strategy must beat.
    """
    if ticker not in prices.columns:
        raise KeyError(
            f"Benchmark ticker {ticker!r} not found in price data columns."
        )
    eval_start = pd.Timestamp(eval_start)
    eval_end = pd.Timestamp(eval_end)
    window = prices.loc[eval_start:eval_end, ticker]
    if window.empty:
        raise ValueError(
            f"No {ticker!r} prices in evaluation window [{eval_start}, {eval_end}]."
        )
    return window.pct_change().fillna(0.0)
