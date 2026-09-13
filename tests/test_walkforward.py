"""Tests for the walk-forward split -- the logic we most need to trust.

These verify that the split boundaries are correct, that tuning and test windows
never overlap, and (critically) that the leakage tripwire rejects returns that
stray into the wrong window.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from invest_tracker.walkforward import LeakageError, Stage, WalkForwardSplit


def _split() -> WalkForwardSplit:
    return WalkForwardSplit(
        period_start=date(2010, 1, 1),
        period_end=date(2024, 12, 31),
        tuning_end=date(2018, 12, 31),
        test_start=date(2019, 1, 1),
        lookback_days=252,
    )


def test_windows_are_correct_and_disjoint():
    s = _split()
    assert s.evaluation_window(Stage.TUNING) == (date(2010, 1, 1), date(2018, 12, 31))
    assert s.evaluation_window(Stage.TEST) == (date(2019, 1, 1), date(2024, 12, 31))

    tune_start, tune_end = s.evaluation_window(Stage.TUNING)
    test_start, test_end = s.evaluation_window(Stage.TEST)
    # No overlap: tuning ends strictly before test begins.
    assert tune_end < test_start


def test_invalid_split_overlap_raises():
    with pytest.raises(LeakageError):
        WalkForwardSplit(
            period_start=date(2010, 1, 1),
            period_end=date(2024, 12, 31),
            tuning_end=date(2019, 6, 30),   # after test_start -> overlap
            test_start=date(2019, 1, 1),
            lookback_days=252,
        )


def test_invalid_split_bounds_raise():
    with pytest.raises(LeakageError):
        WalkForwardSplit(
            period_start=date(2010, 1, 1),
            period_end=date(2024, 12, 31),
            tuning_end=date(2018, 12, 31),
            test_start=date(2025, 6, 1),    # test_start after period_end
            lookback_days=252,
        )


def test_price_window_includes_warmup_before_eval():
    s = _split()
    # Test-window prices must start well before test_start for signal lookback.
    price_start, price_end = s.price_window(Stage.TEST)
    eval_start, eval_end = s.evaluation_window(Stage.TEST)
    assert price_start < eval_start
    assert price_end == eval_end


def test_assert_within_accepts_in_window_returns():
    s = _split()
    idx = pd.bdate_range("2019-02-01", "2019-06-01")  # inside TEST window
    returns = pd.Series(0.001, index=idx)
    # Should pass through unchanged.
    pd.testing.assert_series_equal(s.assert_within(returns, Stage.TEST), returns)


def test_assert_within_rejects_tuning_dates_scored_as_test():
    s = _split()
    # Returns dated in the TUNING window handed to the TEST guard must raise.
    idx = pd.bdate_range("2015-01-01", "2015-03-01")
    returns = pd.Series(0.001, index=idx)
    with pytest.raises(LeakageError):
        s.assert_within(returns, Stage.TEST)


def test_assert_within_rejects_test_dates_scored_as_tuning():
    s = _split()
    idx = pd.bdate_range("2020-01-01", "2020-03-01")  # TEST dates
    returns = pd.Series(0.001, index=idx)
    with pytest.raises(LeakageError):
        s.assert_within(returns, Stage.TUNING)


def test_assert_within_rejects_partial_stray():
    s = _split()
    # Series that straddles the boundary: mostly test, one tuning day.
    idx = pd.DatetimeIndex([date(2018, 12, 31)]).append(
        pd.bdate_range("2019-01-02", "2019-02-01")
    )
    returns = pd.Series(0.001, index=idx)
    with pytest.raises(LeakageError):
        s.assert_within(returns, Stage.TEST)
