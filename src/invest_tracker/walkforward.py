"""Walk-forward split with anti-leakage guardrails.

The whole point of this module is to make it *structurally hard* to score the
strategy on data that was used (or could have been used) for tuning.

Two disjoint windows:

    TUNING : [period_start, tuning_end]   -- parameter selection happens here
    TEST   : [test_start,  period_end]    -- untouched; the reported OOS result

Guardrails:
  * The windows must not overlap (``test_start`` strictly after ``tuning_end``).
  * You never pass raw dates to "score this window" -- you pass a :class:`Stage`.
    The split owns the mapping from stage to dates, so you cannot accidentally
    point the evaluator at the wrong slice.
  * ``evaluation_window`` bounds where *performance* may be measured. A returns
    series handed to :meth:`assert_within` that strays outside the requested
    stage raises -- so scoring on the wrong window fails loudly instead of
    quietly inflating results.
  * ``price_window`` deliberately includes a warm-up buffer *before* each
    evaluation window so signals at the window's first days have full lookback
    history. Using past prices to form a signal is legitimate; measuring returns
    on the wrong window is not, and only the latter is what the guard blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

import pandas as pd


class Stage(str, Enum):
    """Which walk-forward window a run targets."""

    TUNING = "tuning"
    TEST = "test"


class LeakageError(RuntimeError):
    """Raised when an operation would read or score the wrong walk-forward window."""


@dataclass(frozen=True)
class WalkForwardSplit:
    """Owns the tuning/test boundary and enforces the anti-leakage contract."""

    period_start: date
    period_end: date
    tuning_end: date
    test_start: date
    lookback_days: int

    def __post_init__(self) -> None:
        if not (self.period_start <= self.tuning_end < self.test_start <= self.period_end):
            raise LeakageError(
                "Invalid walk-forward split. Require "
                "period_start <= tuning_end < test_start <= period_end. "
                f"Got start={self.period_start}, tuning_end={self.tuning_end}, "
                f"test_start={self.test_start}, end={self.period_end}."
            )
        if self.lookback_days <= 0:
            raise LeakageError("lookback_days must be > 0.")

    # -- Window definitions ---------------------------------------------------

    def evaluation_window(self, stage: Stage) -> tuple[date, date]:
        """The date range over which returns/metrics may be measured for a stage."""
        if stage is Stage.TUNING:
            return (self.period_start, self.tuning_end)
        if stage is Stage.TEST:
            return (self.test_start, self.period_end)
        raise LeakageError(f"Unknown stage: {stage!r}")

    def price_window(
        self, stage: Stage, calendar_buffer_days: int | None = None
    ) -> tuple[date, date]:
        """Date range of *prices* to feed a stage, including a warm-up buffer.

        The buffer extends before the evaluation window so the signal has full
        lookback history on the window's first trading day. It is measured in
        calendar days and defaults to roughly ``2 * lookback_days`` trading days
        converted to calendar days (with slack for weekends/holidays).
        """
        eval_start, eval_end = self.evaluation_window(stage)
        if calendar_buffer_days is None:
            # ~1.5 calendar days per trading day, doubled for safety.
            calendar_buffer_days = int(self.lookback_days * 2 * 1.5) + 10
        warmup_start = eval_start - timedelta(days=calendar_buffer_days)
        return (warmup_start, eval_end)

    # -- Guards ---------------------------------------------------------------

    def assert_within(self, returns: pd.Series, stage: Stage) -> pd.Series:
        """Assert every dated observation lies inside the stage's eval window.

        Returns the (unchanged) series so it can be used inline. Raises
        :class:`LeakageError` if any timestamp falls outside -- the tripwire
        that stops results being computed on the wrong window.
        """
        if returns.empty:
            return returns
        eval_start, eval_end = self.evaluation_window(stage)
        lo, hi = pd.Timestamp(eval_start), pd.Timestamp(eval_end)
        idx = pd.DatetimeIndex(returns.index)
        stray = idx[(idx < lo) | (idx > hi)]
        if len(stray) > 0:
            raise LeakageError(
                f"{len(stray)} observation(s) fall outside the {stage.value} window "
                f"[{eval_start}, {eval_end}]; first stray: {stray[0].date()}. "
                "Refusing to score on the wrong walk-forward window."
            )
        return returns

    def slice_evaluation(self, series_or_frame: pd.Series | pd.DataFrame, stage: Stage):
        """Slice a dated series/frame down to a stage's evaluation window."""
        eval_start, eval_end = self.evaluation_window(stage)
        return series_or_frame.loc[pd.Timestamp(eval_start):pd.Timestamp(eval_end)]


def split_from_spec(spec) -> WalkForwardSplit:
    """Build a :class:`WalkForwardSplit` from a validated :class:`~invest_tracker.config.Spec`."""
    return WalkForwardSplit(
        period_start=spec.period.start,
        period_end=spec.period.end,
        tuning_end=spec.walkforward.tuning_end,
        test_start=spec.walkforward.test_start,
        lookback_days=spec.signal.lookback_days,
    )
