# invest-tracker

A walk-forward backtest harness for a **cross-sectional momentum** strategy on
US equities. This is v1 of an ongoing research project; the priority is **clean
seams and honest methodology over features**.

---

## The strategy spec

The full spec lives in [`configs/momentum_sp500.yaml`](configs/momentum_sp500.yaml)
— strategy parameters are config, never hard-coded in logic.

| Element | v1 setting |
|---|---|
| **Universe** | A liquid US equity cross-section: static S&P 500 snapshot (`data/universe/sp500.csv`). Config-driven, so it can change later. |
| **Signal** | 12-month price momentum skipping the most recent month — return from `t-252` to `t-21` trading days. |
| **Portfolio** | Rank the universe by the signal, go long the **top decile**, **equal-weighted**. |
| **Rebalance** | Monthly (last trading day of each month). |
| **Costs** | Configurable per-trade cost in bps of turnover, charged on every rebalance. Default **10 bps** (conservative). |
| **Benchmark** | SPY buy-and-hold over the same period. |

---

## Methodology constraints (these matter most)

### Walk-forward split
The timeline is divided into two **disjoint** windows:

```
TUNING : [period.start, tuning_end]   -> parameter selection ONLY
TEST   : [test_start,  period.end]    -> untouched; the reported OOS result
```

Leakage is made *structurally hard*, not merely discouraged
([`src/invest_tracker/walkforward.py`](src/invest_tracker/walkforward.py)):

- The windows must not overlap (`test_start` strictly after `tuning_end`); an
  invalid split raises at construction.
- You never hand raw dates to "score this window" — you pass a `Stage`
  (`TUNING`/`TEST`). The split owns the stage→dates mapping, so you can't point
  the evaluator at the wrong slice by accident.
- `assert_within(returns, stage)` is a tripwire: any return dated outside the
  requested stage's window raises `LeakageError` instead of quietly inflating
  results. The runner guards both strategy and benchmark returns this way.
- Price fetches include a warm-up buffer *before* each window so the signal has
  full lookback history on day one. Using past prices to *form* a signal is
  legitimate; measuring *performance* on the wrong window is not — only the
  latter is what the guard blocks.

### Success criterion (explicit, in code)
Defined in [`src/invest_tracker/report.py`](src/invest_tracker/report.py):

> **The strategy must beat SPY buy-and-hold, out-of-sample (the TEST window),
> net of costs** — measured by total return over the window.

Every run prints a single `[PASS]`/`[FAIL]` verdict line so the conclusion
cannot be read two ways.

### Reported metrics
Total return, CAGR, Sharpe (annualized, configurable risk-free rate), and max
drawdown — computed identically for the strategy and SPY and shown side by side.

---

## Architecture

A clean data seam plus one module per concern:

```
src/invest_tracker/
├── config.py           # parse + validate the YAML spec into typed dataclasses
├── data/
│   ├── base.py         # PriceDataSource — the swappable interface (ABC)
│   └── yfinance_source.py  # concrete yfinance impl + on-disk parquet cache
├── universe.py         # config's universe choice -> ticker list
├── signal.py           # pure momentum signal (t-252 -> t-21)
├── costs.py            # turnover-based transaction cost model
├── portfolio.py        # rank -> top-decile equal-weight -> monthly rebalance
├── walkforward.py      # tuning/test split + anti-leakage guardrails
├── metrics.py          # total return, CAGR, Sharpe, max drawdown
├── benchmark.py        # SPY buy-and-hold
├── report.py           # side-by-side results + PASS/FAIL verdict
└── runner.py           # thin orchestration wiring it all together
```

**Swapping the data source** means implementing `PriceDataSource.get_prices`
in a new module and passing it in — no signal/portfolio/walk-forward code
changes. `yfinance_source.py` is the only module that imports yfinance.

---

## Setup

Requires Python 3.10+.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
# or, for an editable install with the `run-backtest` command + dev tools:
pip install -e ".[dev]"
```

The runtime price cache and backtest outputs are gitignored (they are
re-derivable). yfinance requires network access on the first run; subsequent
runs read from `data/cache/`.

---

## Running a backtest

```bash
# Out-of-sample result (default) — this is the honest, reported number:
python scripts/run_backtest.py --config configs/momentum_sp500.yaml --stage test

# Or via the module / installed entry point:
python -m invest_tracker --stage test
run-backtest --stage test          # if installed with pip install -e .

# Parameter-selection window only (never used for the reported result):
python -m invest_tracker --stage tuning

# Both windows, reported separately:
python -m invest_tracker --stage both
```

Example output shape:

```
Walk-forward stage: TEST

Metric              Strategy    SPY (B&H)
------------------------------------------
Total return          xx.xx%       xx.xx%
CAGR                  xx.xx%       xx.xx%
Sharpe                  x.xx         x.xx
Max drawdown         -xx.xx%      -xx.xx%

[PASS] Strategy beats SPY buy-and-hold net of costs on the test window (OUT-OF-SAMPLE).
```

---

## Tests

```bash
pytest
```

- [`tests/test_signal.py`](tests/test_signal.py) — known-answer tests pinning the
  exact `t-252 → t-21` semantics (including off-by-one and the recent-month skip).
- [`tests/test_walkforward.py`](tests/test_walkforward.py) — split boundaries,
  no overlap, and the leakage tripwire rejecting returns scored on the wrong
  window.

---

## Honest v1 limitations

- **Survivorship bias.** The universe is a *current* S&P 500 snapshot applied to
  historical dates, so delisted/acquired names are absent. This biases results
  upward. A point-in-time universe is planned for v2.
- **Turnover approximation.** Weights are assumed reset exactly to target each
  rebalance rather than tracking intra-month drift; for a monthly equal-weight
  large-cap book this slightly *over*states costs (conservative direction).
- **Data.** yfinance adjusted closes fold in dividends/splits and are adequate
  for v1; a higher-quality vendor can be dropped in behind `PriceDataSource`.
- **Fixed parameters.** v1 ships fixed spec parameters; the walk-forward
  scaffolding is built so a genuine tuning search on the TUNING window can be
  added without touching the TEST path.
