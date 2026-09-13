"""Load and validate the YAML backtest spec into typed dataclasses.

Validation is deliberately strict and fails loud: a malformed spec should stop
a run before any data is fetched, not silently produce misleading results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml


@dataclass(frozen=True)
class UniverseConfig:
    source: str
    path: str


@dataclass(frozen=True)
class SignalConfig:
    lookback_days: int
    skip_days: int


@dataclass(frozen=True)
class PortfolioConfig:
    long_quantile: float
    weighting: str
    rebalance: str


@dataclass(frozen=True)
class CostsConfig:
    per_trade_bps: float


@dataclass(frozen=True)
class PeriodConfig:
    start: date
    end: date


@dataclass(frozen=True)
class WalkForwardConfig:
    tuning_end: date
    test_start: date


@dataclass(frozen=True)
class MetricsConfig:
    risk_free_rate: float = 0.0
    periods_per_year: int = 252


@dataclass(frozen=True)
class Spec:
    universe: UniverseConfig
    signal: SignalConfig
    portfolio: PortfolioConfig
    costs: CostsConfig
    benchmark: str
    period: PeriodConfig
    walkforward: WalkForwardConfig
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


class SpecError(ValueError):
    """Raised when the spec file is missing fields or internally inconsistent."""


def _as_date(value: object, field_name: str) -> date:
    """Coerce a YAML scalar to a date. PyYAML parses ISO dates natively, but
    accept strings too so hand-edited specs don't surprise the user."""
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:  # noqa: TRY003 - message is specific
            raise SpecError(f"{field_name!r} is not a valid ISO date: {value!r}") from exc
    raise SpecError(f"{field_name!r} must be a date, got {type(value).__name__}")


def _require(mapping: dict, key: str, context: str) -> object:
    if key not in mapping:
        raise SpecError(f"Missing required key {key!r} in {context}.")
    return mapping[key]


def load_spec(path: str | Path) -> Spec:
    """Parse a YAML spec file into a validated :class:`Spec`."""
    path = Path(path)
    if not path.exists():
        raise SpecError(f"Spec file not found: {path}")

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise SpecError("Spec file must contain a top-level mapping.")

    universe = _require(raw, "universe", "spec")
    signal = _require(raw, "signal", "spec")
    portfolio = _require(raw, "portfolio", "spec")
    costs = _require(raw, "costs", "spec")
    period = _require(raw, "period", "spec")
    walkforward = _require(raw, "walkforward", "spec")
    metrics = raw.get("metrics", {}) or {}

    spec = Spec(
        universe=UniverseConfig(
            source=str(_require(universe, "source", "universe")),
            path=str(_require(universe, "path", "universe")),
        ),
        signal=SignalConfig(
            lookback_days=int(_require(signal, "lookback_days", "signal")),
            skip_days=int(_require(signal, "skip_days", "signal")),
        ),
        portfolio=PortfolioConfig(
            long_quantile=float(_require(portfolio, "long_quantile", "portfolio")),
            weighting=str(_require(portfolio, "weighting", "portfolio")),
            rebalance=str(_require(portfolio, "rebalance", "portfolio")),
        ),
        costs=CostsConfig(
            per_trade_bps=float(_require(costs, "per_trade_bps", "costs")),
        ),
        benchmark=str(_require(raw, "benchmark", "spec")),
        period=PeriodConfig(
            start=_as_date(_require(period, "start", "period"), "period.start"),
            end=_as_date(_require(period, "end", "period"), "period.end"),
        ),
        walkforward=WalkForwardConfig(
            tuning_end=_as_date(
                _require(walkforward, "tuning_end", "walkforward"), "walkforward.tuning_end"
            ),
            test_start=_as_date(
                _require(walkforward, "test_start", "walkforward"), "walkforward.test_start"
            ),
        ),
        metrics=MetricsConfig(
            risk_free_rate=float(metrics.get("risk_free_rate", 0.0)),
            periods_per_year=int(metrics.get("periods_per_year", 252)),
        ),
    )
    _validate(spec)
    return spec


def _validate(spec: Spec) -> None:
    """Cross-field validation. Walk-forward integrity is enforced more deeply in
    walkforward.py; this catches spec-level nonsense early."""
    s = spec.signal
    if s.lookback_days <= 0 or s.skip_days < 0:
        raise SpecError("signal.lookback_days must be > 0 and signal.skip_days >= 0.")
    if s.skip_days >= s.lookback_days:
        raise SpecError("signal.skip_days must be smaller than signal.lookback_days.")

    q = spec.portfolio.long_quantile
    if not 0.0 < q <= 1.0:
        raise SpecError("portfolio.long_quantile must be in (0, 1].")
    if spec.portfolio.weighting != "equal":
        raise SpecError(
            f"Unsupported portfolio.weighting: {spec.portfolio.weighting!r} (v1: 'equal')."
        )
    if spec.portfolio.rebalance != "monthly":
        raise SpecError(
            f"Unsupported portfolio.rebalance: {spec.portfolio.rebalance!r} (v1: 'monthly')."
        )

    if spec.costs.per_trade_bps < 0:
        raise SpecError("costs.per_trade_bps must be >= 0.")

    if spec.universe.source != "csv":
        raise SpecError(f"Unsupported universe.source: {spec.universe.source!r} (v1: 'csv').")

    p, w = spec.period, spec.walkforward
    if p.start >= p.end:
        raise SpecError("period.start must be before period.end.")
    if not (p.start <= w.tuning_end < w.test_start <= p.end):
        raise SpecError(
            "Walk-forward windows are inconsistent. Require: "
            "period.start <= tuning_end < test_start <= period.end. "
            f"Got start={p.start}, tuning_end={w.tuning_end}, "
            f"test_start={w.test_start}, end={p.end}."
        )

    if spec.metrics.periods_per_year <= 0:
        raise SpecError("metrics.periods_per_year must be > 0.")
