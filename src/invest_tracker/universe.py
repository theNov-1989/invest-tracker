"""Universe definition.

Turns the config's universe choice into a list of ticker symbols. v1 reads a
static, committed CSV snapshot. Because the source is config-driven, changing
the universe (a different index, a custom list, a point-in-time file) means
editing the spec, not the strategy.
"""

from __future__ import annotations

from pathlib import Path

from invest_tracker.config import UniverseConfig


def load_universe(cfg: UniverseConfig) -> list[str]:
    """Return the list of tradable tickers for the given universe config."""
    if cfg.source == "csv":
        return _load_csv(cfg.path)
    raise ValueError(f"Unsupported universe source: {cfg.source!r}")


def _load_csv(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")

    tickers: list[str] = []
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower() == "symbol":
            continue
        symbol = line.split(",")[0].strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            tickers.append(symbol)

    if not tickers:
        raise ValueError(f"Universe file {path} contained no tickers.")
    return tickers
