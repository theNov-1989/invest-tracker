"""Data access layer.

Strategy code depends only on the :class:`PriceDataSource` interface in
``base.py``. Concrete implementations (e.g. ``yfinance_source.py``) are the only
place a specific data vendor is imported, so the source can be swapped without
touching the rest of the harness.
"""

from invest_tracker.data.base import PriceDataSource

__all__ = ["PriceDataSource"]
