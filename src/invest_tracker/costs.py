"""Transaction cost model.

Costs are charged on portfolio turnover at each rebalance. Turnover is the sum
of absolute weight changes across all names; a full round-trip of the whole book
(sell everything, buy a fresh book) is turnover of 2.0. The cost is
``turnover * per_trade_bps / 10_000`` as a fraction of portfolio value, deducted
from that rebalance day's return.
"""

from __future__ import annotations

import pandas as pd


def turnover(prev_weights: pd.Series, new_weights: pd.Series) -> float:
    """Sum of absolute weight changes between two weight vectors.

    The two Series need not share an index; missing names are treated as weight
    zero on that side (a new position, or one fully exited).
    """
    aligned_prev, aligned_new = prev_weights.align(new_weights, fill_value=0.0)
    return float((aligned_new - aligned_prev).abs().sum())


def rebalance_cost(
    prev_weights: pd.Series,
    new_weights: pd.Series,
    per_trade_bps: float,
) -> float:
    """Cost (as a fraction of portfolio value) of moving from prev to new weights."""
    if per_trade_bps < 0:
        raise ValueError("per_trade_bps must be >= 0.")
    return turnover(prev_weights, new_weights) * per_trade_bps / 10_000.0
