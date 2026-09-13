"""Concrete :class:`PriceDataSource` backed by yfinance, with an on-disk cache.

This is the only module that imports yfinance. The cache stores fetched price
panels as parquet keyed by the request, so reruns are fast and can work offline
once the data has been pulled. Cached files live under ``data/cache/`` and are
gitignored (they are re-derivable and can be large).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd


class YFinanceSource:
    """Fetch adjusted close prices from Yahoo Finance.

    Implements the :class:`~invest_tracker.data.base.PriceDataSource` contract
    (registered as a virtual subclass below so isinstance checks pass without a
    hard import cycle).
    """

    def __init__(self, cache_dir: str | Path = "data/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, tickers: Sequence[str], start: date, end: date) -> Path:
        key = "|".join(sorted(tickers)) + f"@{start.isoformat()}:{end.isoformat()}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"prices_{digest}.parquet"

    def get_prices(
        self,
        tickers: Sequence[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        cache_path = self._cache_path(tickers, start, end)
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        prices = self._download(tickers, start, end)
        prices.to_parquet(cache_path)
        return prices

    def _download(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        # Imported lazily so the rest of the package (and its tests) does not
        # require yfinance to be installed.
        import yfinance as yf

        # yfinance's `end` is exclusive; add a day to make our contract inclusive.
        end_exclusive = pd.Timestamp(end) + pd.Timedelta(days=1)
        raw = yf.download(
            tickers=list(tickers),
            start=pd.Timestamp(start),
            end=end_exclusive,
            auto_adjust=True,   # 'Close' becomes split/dividend adjusted
            progress=False,
            group_by="column",
        )
        if raw is None or raw.empty:
            raise RuntimeError(
                "yfinance returned no data. Check connectivity and ticker symbols."
            )

        # With multiple tickers yfinance returns columns as a MultiIndex
        # (field, ticker); with one ticker it is flat.
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].copy()
        else:
            close = raw[["Close"]].copy()
            close.columns = [tickers[0]]

        close = close.dropna(axis=1, how="all")   # drop tickers with no data at all
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close = close.sort_index()
        close.index.name = "date"
        return close


# Register as a virtual subclass so `isinstance(src, PriceDataSource)` holds
# without YFinanceSource importing base at class-definition time.
from invest_tracker.data.base import PriceDataSource  # noqa: E402

PriceDataSource.register(YFinanceSource)
