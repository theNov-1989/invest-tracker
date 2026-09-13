"""The data-source seam.

This abstract base class is the *only* contract strategy code knows about. Any
data vendor (yfinance, a paid API, a local database, a CSV dump) can back a
backtest by implementing ``get_prices`` and returning the same shape. Keeping
this interface narrow is what lets the source be swapped later without touching
signal, portfolio, or walk-forward code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date

import pandas as pd


class PriceDataSource(ABC):
    """Interface for a source of daily adjusted close prices."""

    @abstractmethod
    def get_prices(
        self,
        tickers: Sequence[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Return daily adjusted close prices.

        Contract:
          * Return a DataFrame indexed by a sorted, tz-naive ``DatetimeIndex``
            of trading days.
          * Columns are ticker symbols (a subset of ``tickers`` is allowed if
            some have no data; implementations should not invent rows).
          * Values are split/dividend-adjusted close prices as floats.
          * ``start``/``end`` are inclusive calendar bounds.

        Implementations must not look ahead: a price dated ``t`` reflects only
        information available at the close of ``t``.
        """
        raise NotImplementedError
