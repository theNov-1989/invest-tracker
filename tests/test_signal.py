"""Known-answer tests for the momentum signal.

These pin the exact t-252 -> t-21 semantics, including the off-by-one behaviour
that is easy to get wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from invest_tracker.signal import compute_momentum, momentum_as_of


def _price_frame(series: dict[str, list[float]], n: int) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(series, index=idx)


def test_momentum_is_return_from_lookback_to_skip():
    # 400 business days; a single ticker with a known geometric path so we can
    # compute the expected 252->21 return exactly.
    n = 400
    # price[i] = 100 * 1.001**i  -> return over any span depends only on the span.
    prices = _price_frame({"AAA": [100.0 * (1.001 ** i) for i in range(n)]}, n)

    signal = compute_momentum(prices, lookback_days=252, skip_days=21)

    # At position t, signal = price[t-21]/price[t-252] - 1.
    t = 300
    expected = prices["AAA"].iloc[t - 21] / prices["AAA"].iloc[t - 252] - 1.0
    assert signal["AAA"].iloc[t] == pytest.approx(expected)

    # For this constant-growth path the span is (252 - 21) = 231 days.
    assert signal["AAA"].iloc[t] == pytest.approx(1.001 ** 231 - 1.0)


def test_first_lookback_rows_are_nan():
    n = 300
    prices = _price_frame({"AAA": list(np.linspace(100, 200, n))}, n)
    signal = compute_momentum(prices, lookback_days=252, skip_days=21)

    # Before enough history exists the signal must be NaN and thus untradeable.
    assert signal["AAA"].iloc[:252].isna().all()
    assert not np.isnan(signal["AAA"].iloc[252])


def test_skip_window_excludes_recent_month():
    # Ramp up for a year, then crash in the final ~month. With a 21-day skip the
    # signal at the end should reflect the ramp (positive), not the recent crash.
    n = 300
    path = [100.0 + i for i in range(n - 10)] + [50.0] * 10  # crash in last 10 days
    prices = _price_frame({"AAA": path}, n)
    signal = compute_momentum(prices, lookback_days=252, skip_days=21)

    last = signal["AAA"].iloc[-1]
    # t-21 lands before the crash, so momentum stays positive despite the drop.
    assert last > 0


def test_ranking_across_tickers():
    n = 300
    strong = [100.0 * (1.002 ** i) for i in range(n)]   # fast riser
    weak = [100.0 * (1.0005 ** i) for i in range(n)]     # slow riser
    prices = _price_frame({"STRONG": strong, "WEAK": weak}, n)

    row = momentum_as_of(prices, prices.index[-1], lookback_days=252, skip_days=21)
    assert row["STRONG"] > row["WEAK"]


def test_invalid_params_raise():
    prices = _price_frame({"AAA": [1.0, 2.0, 3.0]}, 3)
    with pytest.raises(ValueError):
        compute_momentum(prices, lookback_days=0, skip_days=0)
    with pytest.raises(ValueError):
        compute_momentum(prices, lookback_days=10, skip_days=10)  # skip >= lookback
