"""invest-tracker: a walk-forward cross-sectional momentum backtest harness.

The package is organized into single-responsibility modules connected by clean
seams:

    config      -- parse + validate the YAML spec into typed dataclasses
    data/       -- price data behind a swappable PriceDataSource interface
    universe    -- turn the config's universe choice into a ticker list
    signal      -- pure momentum signal computation
    costs       -- turnover-based transaction cost model
    portfolio   -- ranking, top-decile equal-weight, monthly rebalance engine
    walkforward -- tuning/test split with anti-leakage guardrails
    metrics     -- total return, CAGR, Sharpe, max drawdown
    benchmark   -- SPY buy-and-hold
    report      -- side-by-side results and the explicit PASS/FAIL verdict
"""

__version__ = "0.1.0"
