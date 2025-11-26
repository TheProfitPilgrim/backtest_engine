# backtest_engine

A small, config‑driven backtest engine for Indian mutual funds, designed to be reusable, **no‑lookahead**, and easy to drive from notebooks or a future UI.

The long‑term goal is to replace one‑off research notebooks with a clean Python library that:

- Talks to the existing Postgres DB (ACE feed + internal tables)
- Takes a **BacktestConfig** (matching Notion “Variant” fields)
- Runs backtests with well‑defined rules (universe, signal, selection, rebalance)
- Compares against a benchmark (currently Nifty 500 TRI)
- Returns structured outputs that can be saved and analysed elsewhere

This README describes the **current state** of the engine (Phase 0/1). It will evolve as we add more features (rolling portfolios, SIPs, tax, etc.).

---

## 1. Repository layout

Current tree (simplified):

```text
backtest_engine/
├── notebooks/
│   └── 01_smoke_test.ipynb      # end‑to‑end sanity checks / examples
├── requirements.txt
├── pyproject.toml               # package metadata (editable install)
├── README.md                    # (you can replace this with this file)
└── src/
    └── backtest_engine/
        ├── __init__.py
        ├── config.py            # BacktestConfig and related dataclasses
        ├── data_provider.py     # DataProvider interface
        ├── db.py                # SQLAlchemy engine + run_sql helper
        ├── engine.py            # BacktestEngine core loop
        ├── postgres_provider.py # PostgresDataProvider implementation
        ├── utils/
        │   └── dates.py         # generate_rebalance_dates helper
        └── data/
            └── market/
                ├── nifty500_raw.csv  # raw download from Investing.com
                └── nifty500_tri.csv  # cleaned benchmark series
```

The engine is written as a normal Python package under `src/backtest_engine/` so it can be installed in editable mode and reused from notebooks or (later) a UI/backend.

---

## 2. Environment & installation

### 2.1. Create virtual environment (VS Code or terminal)

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate     # or .venv\Scripts\activate on Windows
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Editable install of the package itself
pip install -e .
```

VS Code is already configured to create and use `.venv`; once that’s working, the engine can be imported as:

```python
from backtest_engine import BacktestEngine
```

### 2.2. Postgres tunnel & environment variables

The engine assumes you can already connect to the existing Kairo Postgres instance via SSH tunnel.

Typical setup (from terminal, *before* running notebooks):

```bash
export VPS_HOST=206.189.136.217
export DB_HOST=private-kairo-postgresql-blr1-50081-do-user-13709976-0.l.db.ondigitalocean.com
export DB_PORT=25060
export LOCAL_PORT=5433
export SSH_KEY=/Users/sanjay/.ssh/id_rsa.pub

./scripts/tunnel.sh start
# Enter SSH username when prompted (e.g. "sid")
```

Then export DB credentials used by `backtest_engine.db.get_engine`:

```bash
export PGHOST=localhost
export PGPORT=5433
export PGDATABASE=kairo_production
export PGUSER=sanjay_readonly
export PGPASSWORD='Piper112358!'   # or set this in your shell startup
```

If `PGPASSWORD` is not set, the engine will raise a clear `RuntimeError` when trying to connect.

---

## 3. Data providers and no‑lookahead design

The engine uses a **DataProvider** abstraction so the core logic doesn’t depend on where data comes from.

### 3.1. DataProvider interface

`data_provider.py` defines an abstract interface:

- `get_universe(as_of: date, universe_config: UniverseConfig) -> pd.DataFrame`
- `get_signal_scores(as_of: date, schemecodes: Iterable[int], signal_config: SignalConfig) -> pd.DataFrame`
- `get_nav_series(schemecodes: list[int], start: date, end: date) -> pd.DataFrame`
- `get_benchmark_series(start: date, end: date) -> pd.DataFrame`

### 3.2. PostgresDataProvider (current implementation)

`postgres_provider.py` implements `DataProvider` against the existing Kairo DB:

- **Universe** uses `scheme_details`, `sclass_mst`, `plan_mst`, etc.
- **Signals** use `performance_ranking` (plan_code = 5, direct plan).
- **NAVs** use `navhist`.
- **Benchmark** loads Nifty 500 TRI from a local CSV.

The key no‑lookahead invariant is in `get_signal_scores`:

- For each fund (`schemecode`), it selects the **latest** row from `performance_ranking` with
  `pr.date <= as_of` (no future data).

So when we rebalance on, say, 2018‑04‑01, the signal only sees rankings/returns that were already known on or before that date.

---

## 4. Configuration model (BacktestConfig)

The engine is driven entirely by a `BacktestConfig` dataclass in `config.py`. This is intended to map 1‑to‑1 with a “Variant” row in Notion.

### 4.1. Study window

```python
@dataclass
class StudyWindow:
    start: date
    end: date
```

Defines the time range for the backtest. The engine:

- Generates rebalance dates inside `[start, end)`
- Computes portfolio and benchmark returns over this window

### 4.2. UniverseConfig

Right now we have a simple preset‑based universe:

```python
@dataclass
class UniverseConfig:
    preset: str  # e.g. "equity_active_direct"
```

The only implemented preset so far is:

- `"equity_active_direct"`: equity funds, active (non‑index), direct plan, growth, active & investible today, excluding liquid/overnight/thematic/sector/etc. (the exact SQL lives in `PostgresDataProvider.get_universe`).

Later this will be extended with extra filters like min AUM, min age, include/exclude category lists, etc.

### 4.3. SignalConfig

Signals control how we rank funds. Currently:

```python
@dataclass
class SignalConfig:
    name: str                       # e.g. "rank_12m_category"
    source: str = "performance_ranking"
    lookback_months: int = 12
    direction: Literal["asc", "desc"] = "desc"
    rank_scope: Literal["category", "asset_class", "universe"] = "category"
    tie_breaker: Optional[str] = None

    # New (optional) fields:
    expression: Optional[str] = None
    filter_expression: Optional[str] = None
```

Two usage modes:

1. **Simple column mode (default)**  
   Leave `expression` and `filter_expression` as `None`.  
   The engine resolves `name`, `lookback_months`, `rank_scope` into a real column in `performance_ranking`
   (e.g. `rank_12m_category`) and uses that as the score.

2. **Expression mode (experimental)**  
   Set `expression` to a small formula using columns from `performance_ranking`, e.g.:

   ```python
   SignalConfig(
       name="mom_1y_3y_blend",
       direction="desc",
       expression="0.5 * perf_1y + 0.5 * perf_3y",
       filter_expression="aum_cr > 300 and age_years > 3",
   )
   ```

   - `filter_expression` is applied first (boolean mask).
   - `expression` is evaluated with pandas on the snapshot to produce a `score` column.
   - `direction` controls whether lower or higher scores are better.

### 4.4. SelectionConfig

Controls how many funds go into the portfolio and how they’re weighted:

```python
@dataclass
class SelectionConfig:
    mode: Literal["top_n", "all"] = "top_n"
    top_n: int = 15
    min_funds: int = 10
    weight_scheme: Literal["equal"] = "equal"
```

- `mode="top_n"`: pick the best `top_n` funds by score (current default).
- `mode="all"`: pick **all** eligible funds (ignoring `top_n`), but still enforce `min_funds`.
- `weight_scheme="equal"`: equal‑weight across the selected funds (1/N). More schemes will be added later.

If the number of eligible funds on a rebalance date is below `min_funds`, the engine raises a `RuntimeError` rather than silently running a tiny portfolio.

### 4.5. RebalanceConfig

Controls how often the portfolio is refreshed:

```python
@dataclass
class RebalanceConfig:
    frequency: str  # "NONE", "3M", "6M", "12M", "18M", "24M"
```

- `"NONE"`: single buy‑and‑hold from `study_window.start` to `study_window.end`.
- `"3M"`, `"6M"`, `"12M"`, `"18M"`, `"24M"`: calendar‑based rebalancing as per `generate_rebalance_dates`.

Currently the engine supports **one portfolio** born at `study_window.start`, with optional rebalancing inside the window. Rolling multi‑portfolio cohorts are planned for later.

### 4.6. BacktestConfig

Putting it all together:

```python
@dataclass
class BacktestConfig:
    name: str
    study_window: StudyWindow
    universe: UniverseConfig
    signal: SignalConfig
    selection: SelectionConfig
    rebalance: RebalanceConfig
```

You create a `BacktestConfig` in a notebook and pass it to `BacktestEngine.run`.

---

## 5. BacktestEngine: what it does today

`engine.py` contains `BacktestEngine`, the core orchestrator.

### 5.1. High‑level algorithm

For a given `BacktestConfig` and `DataProvider`:

1. Generate rebalance dates inside the study window.
2. For each rebalance date:
   1. Get the investible universe as of that date (`get_universe`).
   2. Get signal scores with no lookahead (`get_signal_scores`):
      - latest `performance_ranking` row per fund with `date <= rebalance_date`.
   3. Sort funds by score and apply `SelectionConfig`:
      - `mode="top_n"`: take `top_n`
      - `mode="all"`: take all eligible
   4. Equal‑weight the portfolio.
   5. Pull NAV history for those funds over `[period_start, period_end]` (`get_nav_series`).
   6. Compute fund‑level total returns in the period.
   7. Compute portfolio‑level gross return and period CAGR.
   8. Pull benchmark series over the same dates and compute benchmark return and CAGR.
   9. Update compounded portfolio and benchmark equity curves.
   10. Store:
       - Period‑level summary row.
       - Fund‑level holdings with weights, returns, and period metrics.
3. At the end:
   - Compute total gross return & CAGR for the portfolio and benchmark.
   - Compute total alpha (return and CAGR).
   - Build:
     - `summary` (one row per run),
     - `portfolio_periods` (one row per rebalance period),
     - `holdings` (one row per fund‑period).

The result is wrapped in a simple `BacktestResult` object:

```python
class BacktestResult:
    def __init__(self, run_id, summary, portfolio_periods, holdings):
        self.run_id = run_id
        self.summary = summary
        self.portfolio_periods = portfolio_periods
        self.holdings = holdings
```

Later we’ll add convenience methods like `save(out_dir, level)` to write CSVs.

### 5.2. Benchmark support (Nifty 500 TRI)

For now, the benchmark is hard‑wired as **Nifty 500 TRI**:

- A raw CSV from Investing.com is cleaned into `data/market/nifty500_tri.csv` with two columns:

  ```text
  date,value
  ```

- `PostgresDataProvider.get_benchmark_series(start, end)`:
  - loads and caches this CSV,
  - filters rows between `start` and `end`,
  - returns a DataFrame with `date` and `value`.

The engine then:

- Computes benchmark return per period (end / start − 1),
- Computes benchmark CAGR per period,
- Tracks a benchmark equity curve over the whole backtest,
- Computes period‑level and total alpha (portfolio − benchmark).

In future, the benchmark will become configurable (other indices, index funds, DB tables, etc.).

---

## 6. Example usage (smoke test)

In `notebooks/01_smoke_test.ipynb` you can run a simple 10‑year backtest:

```python
from datetime import date

from backtest_engine import BacktestEngine
from backtest_engine.config import (
    BacktestConfig,
    StudyWindow,
    UniverseConfig,
    SignalConfig,
    SelectionConfig,
    RebalanceConfig,
)
from backtest_engine.postgres_provider import PostgresDataProvider

config = BacktestConfig(
    name="mf-10y-annual-rebal",
    study_window=StudyWindow(start=date(2014, 1, 1), end=date(2024, 1, 1)),
    universe=UniverseConfig(preset="equity_active_direct"),
    signal=SignalConfig(
        name="rank_12m_category",
        direction="asc",            # lower rank = better
    ),
    selection=SelectionConfig(
        mode="top_n",
        top_n=15,
        min_funds=10,
        weight_scheme="equal",
    ),
    rebalance=RebalanceConfig(frequency="12M"),  # annual rebalancing
)

provider = PostgresDataProvider()
engine = BacktestEngine(provider)

result = engine.run(config)

result.summary
result.portfolio_periods.head()
result.holdings.head()
```

You should see:

- `summary`: one row with total gross/benchmark returns & CGARs and alpha.
- `portfolio_periods`: one row per rebalance period (roughly 10 for a 10‑year annual backtest).
- `holdings`: fund‑level detail per period with weights and returns.

---

## 7. Limitations & roadmap (as of now)

What works today:

- ✅ Single‑portfolio backtests with:
  - configurable study window,
  - calendar‑based rebalancing,
  - universe preset `"equity_active_direct"`,
  - rank‑based signals from `performance_ranking` (no lookahead),
  - selection modes `"top_n"` and `"all"`,
  - equal‑weight portfolios,
  - Nifty 500 TRI benchmark comparison.
- ✅ Clean separation between:
  - configuration,
  - data access (DataProvider),
  - core engine loop.
- ✅ Early support for expression‑based signals (formula + filter) on top of `performance_ranking`.

Not implemented yet (planned):

- ⏳ Multi‑portfolio / rolling cohorts (e.g. new portfolio every month/year).
- ⏳ SIP / cashflow patterns (periodic contributions, withdrawals).
- ⏳ Tax module (STCG/LTCG on realised trades).
- ⏳ More flexible universe filters (min AUM, age, include/exclude categories, custom lists).
- ⏳ Alternative benchmarks (other indices, index funds, DB‑based benchmarks).
- ⏳ Output helpers to save results to CSV at different detail levels.
- ⏳ A thin UI layer that wraps BacktestConfig into a user‑friendly form.

This README is intentionally scoped to **what exists right now**. As the engine grows (more configs, tax, SIPs, rolling cohorts), we can expand this document and keep it as the main onboarding reference for both devs and non‑tech users who just want to run backtests via a UI.

---

## 8. Philosophy

A few core principles driving this design:

- **No lookahead by construction** – all signals are computed using data with `date <= rebalance_date`.
- **Config‑first** – everything that matters is encoded in `BacktestConfig`, not hidden in code.
- **Separation of concerns** – data access vs backtest logic vs outputs vs UI.
- **Start thin, then deepen** – build a small, trustworthy slice end‑to‑end, validate against existing notebooks, then add complexity in layers.

If you’re reading this from the future and things are more complex now, hopefully this gives you a clear picture of the foundations the engine started from.