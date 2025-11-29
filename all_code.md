## all_code.py

```python
import os
import json

# Function to extract the Python code from a .py or .ipynb file
def extract_code_from_file(file_path):
    if file_path.endswith('.py'):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    elif file_path.endswith('.ipynb'):
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        code_cells = []
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                code = ''.join(cell.get('source', []))
                code_cells.append(code)

        return "\n\n".join(code_cells)

    return ""


# Function to create the Markdown content
def create_markdown_from_code(root_dir, output_md):
    with open(output_md, 'w', encoding='utf-8') as md_file:
        for root, dirs, files in os.walk(root_dir):

            # Skip .venv entirely
            if '.venv' in dirs:
                dirs.remove('.venv')

            for file in files:
                if file.endswith(('.py', '.ipynb')):  # Include .py and .ipynb
                    file_path = os.path.join(root, file)
                    code = extract_code_from_file(file_path)

                    if not code.strip():
                        continue  # Skip empty results (e.g., ipynb with no code cells)

                    # Write the title (file name) and the code to the markdown file
                    md_file.write(f"## {file}\n\n")
                    md_file.write("```python\n")
                    md_file.write(code)
                    md_file.write("\n```\n\n")


# Define the root directory and the output markdown file
root_directory = '.'  # Current directory (change this if needed)
output_markdown = 'all_code.md'

# Create the markdown file
create_markdown_from_code(root_directory, output_markdown)

print(f"Markdown file '{output_markdown}' has been created with all code.")
```

## streamlit_app.py

```python
from __future__ import annotations

from datetime import date
from pathlib import Path
import os
from typing import Optional

import pandas as pd
import streamlit as st

from backtest_engine import BacktestEngine, PostgresDataProvider, run_batch
from backtest_engine.config import (
    BacktestConfig,
    StudyWindow,
    UniverseConfig,
    SignalConfig,
    SelectionConfig,
    RebalanceConfig,
    FeeConfig,
)

# Root folder for saved runs (relative to repo root)
BACKTEST_ROOT = Path("backtests")

# Default DB env vars (for local development)
DEFAULT_DB_ENV = {
    "PGHOST": "localhost",
    "PGPORT": "5433",
    "PGDATABASE": "kairo_production",
    "PGUSER": "sanjay_readonly",
    "PGPASSWORD": "Piper112358!",
}

# Set defaults only if not already set in the environment
for key, value in DEFAULT_DB_ENV.items():
    if not os.getenv(key):
        os.environ[key] = value

def make_config(
    name: str,
    start: date,
    end: date,
    universe_preset: str,
    signal_mode: str,
    signal_name: str,
    signal_direction: str,
    expression: Optional[str],
    filter_expression: Optional[str],
    selection_mode: str,
    top_n: int,
    min_funds: int,
    weight_scheme: str,
    rebalance_freq: str,
    apply_fee: bool,
    fee_bps: float,
) -> BacktestConfig:
    """Build a BacktestConfig from UI inputs."""
    # Build SignalConfig depending on mode
    if signal_mode == "simple":
        signal_cfg = SignalConfig(
            name=signal_name,
            direction=signal_direction,
        )
    else:
        signal_cfg = SignalConfig(
            name="expression",
            expression=expression or "",
            filter_expression=filter_expression or "",
            direction=signal_direction,
        )

    selection_cfg = SelectionConfig(
        mode=selection_mode,
        top_n=top_n,
        min_funds=min_funds,
        weight_scheme=weight_scheme,
    )

    cfg = BacktestConfig(
        name=name,
        study_window=StudyWindow(start=start, end=end),
        universe=UniverseConfig(preset=universe_preset),
        signal=signal_cfg,
        selection=selection_cfg,
        rebalance=RebalanceConfig(frequency=rebalance_freq),
        fees=FeeConfig(
            apply=apply_fee,
            annual_bps=fee_bps,
        ),
    )
    return cfg


def save_to_index(result_summary: pd.DataFrame, run_dir: Path, save_level: str) -> None:
    """Append this run to backtests/index.csv for easy browsing later."""
    BACKTEST_ROOT.mkdir(parents=True, exist_ok=True)
    index_path = BACKTEST_ROOT / "index.csv"

    summary = result_summary.copy()
    summary["output_dir"] = str(run_dir)
    summary["save_level"] = save_level

    if index_path.exists():
        existing = pd.read_csv(index_path)
        combined = pd.concat([existing, summary], ignore_index=True)
    else:
        combined = summary

    combined.to_csv(index_path, index=False)


def main() -> None:
    st.set_page_config(
        page_title="MF Backtest Engine",
        layout="wide",
    )

    st.title("📈 Mutual Fund Backtest Engine")

    st.markdown(
        "This UI wraps the `backtest_engine` package.\n\n"
        "Make sure your SSH tunnel to the Kairo DB is running and PG* env vars are set "
        "before running backtests."
    )

    tabs = st.tabs(["Run backtest", "Saved backtests"])

    # -------------------------------------------------------------------------
    # TAB 1: RUN BACKTEST
    # -------------------------------------------------------------------------
    with tabs[0]:
        st.header("Run a new backtest")

        col_run_left, col_run_right = st.columns(2)

        with col_run_left:
            st.subheader("Universe & dates")

            run_name = st.text_input(
                "Run name",
                value="mf-10y-annual-top15-net1%",
                help="Used for folder names and file prefixes.",
            )

            universe_preset = st.selectbox(
                "Universe preset",
                options=["equity_active_direct"],
                index=0,
                help="More presets can be added later.",
            )

            start_date = st.date_input(
                "Start date",
                value=date(2014, 1, 1),
            )
            end_date = st.date_input(
                "End date",
                value=date(2024, 1, 1),
            )

            if start_date >= end_date:
                st.error("Start date must be before end date.")

            st.subheader("Rebalancing")

            rebalance_freq = st.selectbox(
                "Rebalance frequency",
                options=["NONE", "3M", "6M", "12M", "18M", "24M"],
                index=3,  # default 12M
            )

            st.subheader("Fees")

            apply_fee = st.checkbox(
                "Apply annual fee?",
                value=True,
            )
            fee_bps = st.slider(
                "Fee (bps per year)",
                min_value=0,
                max_value=300,
                value=100,
                step=10,
            )

            save_level = st.selectbox(
                "Output detail level",
                options=["light", "standard", "full"],
                index=1,
                help="Full includes holdings & config; can be large.",
            )

        with col_run_right:
            st.subheader("Signal (selection criteria)")

            signal_mode = st.radio(
                "Signal mode",
                options=[
                    "simple",
                    "expression",
                ],
                format_func=lambda x: "Simple column" if x == "simple" else "Expression (formula-based)",
            )

            if signal_mode == "simple":
                signal_name = st.text_input(
                    "Signal column name",
                    value="rank_12m_category",
                    help="Must correspond to a column in performance_ranking.",
                )
                expression = None
                filter_expression = None
            else:
                signal_name = "expression"
                expression = st.text_area(
                    "Score expression",
                    value="perf_1y * 0.5 + perf_3y * 0.5",
                    height=80,
                    help="Use columns from performance_ranking (e.g. perf_1y, perf_3y, aum_cr, etc.).",
                )
                filter_expression = st.text_area(
                    "Filter expression (optional)",
                    value="aum_cr > 300 and age_years >= 3",
                    height=60,
                    help="Only funds where this is True will be eligible.",
                )

            signal_direction_label = st.radio(
                "Signal direction",
                options=["asc", "desc"],
                format_func=lambda x: "Ascending (lower score is better)" if x == "asc" else "Descending (higher is better)",
                index=0,
            )
            signal_direction = signal_direction_label  # already 'asc' or 'desc'

            st.subheader("Selection")

            selection_mode = st.radio(
                "Selection mode",
                options=["top_n", "all"],
                format_func=lambda x: "Top N" if x == "top_n" else "All eligible",
            )

            if selection_mode == "top_n":
                top_n = st.slider(
                    "Top N funds",
                    min_value=1,
                    max_value=100,
                    value=15,
                    step=1,
                )
            else:
                top_n = 999999  # effectively "all", but SelectionConfig will only use it for top_n mode

            min_funds = st.slider(
                "Minimum number of funds required",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                help="If fewer eligible funds are found, the run will error out.",
            )

            weight_scheme = "equal"  # only equal-weight implemented for now
            st.markdown("Weighting scheme: **equal-weight** (1/N) for now.")

        run_button = st.button("🚀 Run backtest", type="primary")

        if run_button:
            with st.spinner("Running backtest..."):
                try:
                    provider = PostgresDataProvider()

                    cfg = make_config(
                        name=run_name,
                        start=start_date,
                        end=end_date,
                        universe_preset=universe_preset,
                        signal_mode=signal_mode,
                        signal_name=signal_name,
                        signal_direction=signal_direction,
                        expression=expression,
                        filter_expression=filter_expression,
                        selection_mode=selection_mode,
                        top_n=top_n,
                        min_funds=min_funds,
                        weight_scheme=weight_scheme,
                        rebalance_freq=rebalance_freq,
                        apply_fee=apply_fee,
                        fee_bps=float(fee_bps),
                    )

                    engine = BacktestEngine(provider)
                    result = engine.run(cfg)

                    # Save outputs under backtests/<run_name>/
                    run_dir = BACKTEST_ROOT / run_name
                    run_dir.mkdir(parents=True, exist_ok=True)
                    paths = result.save(run_dir, level=save_level)

                    # Update index
                    save_to_index(result.summary, run_dir, save_level)

                    st.success("Backtest completed ✅")
                    st.subheader("Summary")
                    st.dataframe(result.summary)

                    st.subheader("Period-level results (first 20 rows)")
                    if "period_no" in result.portfolio_periods.columns:
                        st.dataframe(
                            result.portfolio_periods.head(20)
                        )
                    else:
                        st.write("No period-level data available.")

                    st.subheader("Saved files")
                    for key, path in paths.items():
                        st.write(f"- **{key}** → `{path}`")

                except Exception as e:
                    st.error(f"Backtest failed: {e!r}")

    # -------------------------------------------------------------------------
    # TAB 2: SAVED BACKTESTS
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.header("Saved backtests")

        index_path = BACKTEST_ROOT / "index.csv"
        if not index_path.exists():
            st.info("No saved backtests yet. Run a backtest in the first tab.")
            return

        index_df = pd.read_csv(index_path)

        if index_df.empty:
            st.info("Index is empty. Run a backtest in the first tab.")
            return

        st.subheader("Backtest index")
        st.dataframe(index_df)

        run_names = index_df["run_id"].unique().tolist() if "run_id" in index_df.columns else []

        if run_names:
            selected_run = st.selectbox(
                "Select a run to inspect",
                options=run_names,
            )
            if selected_run:
                # Find row for this run
                row = index_df[index_df["run_id"] == selected_run].iloc[0]
                run_dir = Path(row["output_dir"])

                st.markdown(f"### Details for **{selected_run}**")
                st.write(f"Output directory: `{run_dir}`")

                # Try loading summary / periods / holdings
                summary_path = run_dir / f"{selected_run}.summary.csv"
                periods_path = run_dir / f"{selected_run}.periods.csv"
                holdings_path = run_dir / f"{selected_run}.holdings.csv"

                if summary_path.exists():
                    st.subheader("Summary")
                    st.dataframe(pd.read_csv(summary_path))

                if periods_path.exists():
                    st.subheader("Period-level results")
                    st.dataframe(pd.read_csv(periods_path).head(50))

                if holdings_path.exists():
                    with st.expander("Holdings (first 100 rows)"):
                        st.dataframe(pd.read_csv(holdings_path).head(100))
        else:
            st.info("No run_id column found in index.csv; cannot select runs.")


if __name__ == "__main__":
    main()

```

## 02_frontend.ipynb

```python
# 02_frontend.ipynb
# "Mini front end" for running backtests with simple inputs

from datetime import date
from pathlib import Path
import os

import pandas as pd

from backtest_engine import BacktestEngine, PostgresDataProvider, run_batch
from backtest_engine.config import (
    BacktestConfig,
    StudyWindow,
    UniverseConfig,
    SignalConfig,
    SelectionConfig,
    RebalanceConfig,
    FeeConfig,
)

pd.options.display.max_rows = 20
pd.options.display.max_columns = None
pd.options.display.width = 160

print("✅ Imports OK")


# OPTIONAL: Set PG* env vars here if they are missing.
# Comment out or edit as you like.

defaults = {
    "PGHOST": "localhost",
    "PGPORT": "5433",
    "PGDATABASE": "kairo_production",
    "PGUSER": "sanjay_readonly",
    "PGPASSWORD": "Piper112358!",
}

for key, value in defaults.items():
    if not os.getenv(key):
        os.environ[key] = value

print("PGHOST =", os.getenv("PGHOST"))
print("PGPORT =", os.getenv("PGPORT"))
print("PGDATABASE =", os.getenv("PGDATABASE"))
print("PGUSER =", os.getenv("PGUSER"))
print("✅ PG* env vars are set (assuming SSH tunnel is running)")


from backtest_engine.db import run_sql

provider = PostgresDataProvider()

try:
    df_now = run_sql("SELECT now() AT TIME ZONE 'Asia/Kolkata' AS ist_now;")
    display(df_now)
    print("✅ DB connected and responding")
except Exception as e:
    print("❌ DB connectivity issue:", repr(e))
    print("Check SSH tunnel + PG* environment variables.")


def make_config(
    name: str,
    start: date,
    end: date,
    rebalance_freq: str = "12M",
    universe_preset: str = "equity_active_direct",
    signal_name: str = "rank_12m_category",
    signal_direction: str = "asc",
    selection_mode: str = "top_n",
    top_n: int = 15,
    min_funds: int = 10,
    weight_scheme: str = "equal",
    fee_bps: float = 0.0,
    apply_fee: bool = False,
) -> BacktestConfig:
    """Convenience builder for BacktestConfig.

    This is where we centralise sensible defaults for the 'frontend'.
    """
    return BacktestConfig(
        name=name,
        study_window=StudyWindow(start=start, end=end),
        universe=UniverseConfig(preset=universe_preset),
        signal=SignalConfig(
            name=signal_name,
            direction=signal_direction,  # "asc" -> lower score is better, for ranks
        ),
        selection=SelectionConfig(
            mode=selection_mode,        # "top_n" or "all"
            top_n=top_n,
            min_funds=min_funds,
            weight_scheme=weight_scheme,
        ),
        rebalance=RebalanceConfig(
            frequency=rebalance_freq,   # "NONE", "3M", "6M", "12M", "18M", "24M"
        ),
        fees=FeeConfig(
            apply=apply_fee,
            annual_bps=fee_bps,
        ),
    )


# === SINGLE RUN INPUTS ===
# Edit these values and then run this cell + the next one.

RUN_NAME = "mf-10y-annual-top15-net1%"   # used in filenames, folder names

START_DATE = date(2014, 1, 1)
END_DATE   = date(2024, 1, 1)

UNIVERSE_PRESET = "equity_active_direct"

SIGNAL_NAME       = "rank_12m_category"
SIGNAL_DIRECTION  = "asc"    # "asc" for rank columns (1 = best), "desc" for perf columns

SELECTION_MODE = "top_n"     # "top_n" or "all"
TOP_N          = 15
MIN_FUNDS      = 10
WEIGHT_SCHEME  = "equal"     # only "equal" is implemented

REBALANCE_FREQ = "12M"       # "NONE", "3M", "6M", "12M", "18M", "24M"

APPLY_FEE = True
FEE_BPS   = 100.0            # 100 bps = 1% p.a.

# Where to save this run's outputs
SINGLE_RUN_OUTDIR = Path("outputs/single_runs")

# Output detail level: "light", "standard", "full"
SAVE_LEVEL = "standard"

print("Configured single run:", RUN_NAME)


# Run a single backtest based on the form above

engine = BacktestEngine(provider)

single_config = make_config(
    name=RUN_NAME,
    start=START_DATE,
    end=END_DATE,
    rebalance_freq=REBALANCE_FREQ,
    universe_preset=UNIVERSE_PRESET,
    signal_name=SIGNAL_NAME,
    signal_direction=SIGNAL_DIRECTION,
    selection_mode=SELECTION_MODE,
    top_n=TOP_N,
    min_funds=MIN_FUNDS,
    weight_scheme=WEIGHT_SCHEME,
    fee_bps=FEE_BPS,
    apply_fee=APPLY_FEE,
)

print("Running config:")
display(single_config)

result = engine.run(single_config)
print("✅ Backtest completed")

display(result.summary)


# Period-level summary (per rebalance period)

cols_periods = [
    "period_no",
    "start_date",
    "end_date",
    "period_days",
    "num_funds",
    "gross_return",
    "net_return",
    "benchmark_return",
    "alpha_return",
    "net_alpha_return",
]

display(result.portfolio_periods[cols_periods].head(20))

# Optional: holdings sample (first 20 rows)

cols_holdings = [
    "run_id",
    "period_no",
    "rebalance_date",
    "schemecode",
    "scheme_name",
    "weight",
    "fund_return",
    "period_gross_return",
    "period_net_return",
]

display(result.holdings[cols_holdings].head(20))


SINGLE_RUN_OUTDIR.mkdir(parents=True, exist_ok=True)

paths_single = result.save(SINGLE_RUN_OUTDIR, level=SAVE_LEVEL)
print("Saved files:")
for key, path in paths_single.items():
    print(f"  {key}: {path}")


# === BATCH RUN EXAMPLE ===
# Define multiple configs (variants) and run them all in one go.

BATCH_OUTDIR = Path("batch_outputs/frontend_examples")

base_window = StudyWindow(start=date(2014, 1, 1), end=date(2024, 1, 1))

batch_configs = [
    make_config(
        name="mf-10y-annual-top15-net1%",
        start=base_window.start,
        end=base_window.end,
        rebalance_freq="12M",
        top_n=15,
        min_funds=10,
        fee_bps=100.0,
        apply_fee=True,
    ),
    make_config(
        name="mf-10y-annual-top30-net1%",
        start=base_window.start,
        end=base_window.end,
        rebalance_freq="12M",
        top_n=30,
        min_funds=20,
        fee_bps=100.0,
        apply_fee=True,
    ),
    make_config(
        name="mf-10y-semiannual-top15-net1%",
        start=base_window.start,
        end=base_window.end,
        rebalance_freq="6M",
        top_n=15,
        min_funds=10,
        fee_bps=100.0,
        apply_fee=True,
    ),
]

print("Batch configs:")
for cfg in batch_configs:
    print(" -", cfg.name)


batch_summary = run_batch(
    configs=batch_configs,
    data_provider=provider,
    out_dir=BATCH_OUTDIR,
    level="standard",  # or "light" / "full"
)

print("✅ Batch completed")
display(batch_summary)

```

## 01_smoke_test.ipynb

```python
# 01_smoke_test.ipynb
# End-to-end sanity check for backtest_engine

from datetime import date
from pathlib import Path

import pandas as pd

from backtest_engine import BacktestEngine
from backtest_engine.config import (
    BacktestConfig,
    StudyWindow,
    UniverseConfig,
    SignalConfig,
    SelectionConfig,
    RebalanceConfig,
    FeeConfig,
)
from backtest_engine.postgres_provider import PostgresDataProvider
from backtest_engine.db import run_sql

pd.options.display.max_rows = 20
pd.options.display.max_columns = None
pd.options.display.width = 140

print("✅ Imports OK")

import os

# Your usual DB settings
os.environ["PGHOST"] = "localhost"
os.environ["PGPORT"] = "5433"
os.environ["PGDATABASE"] = "kairo_production"
os.environ["PGUSER"] = "sanjay_readonly"
os.environ["PGPASSWORD"] = "Piper112358!"

# Quick DB check – requires tunnel + env vars set in terminal
# (PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD)

try:
    df_now = run_sql("SELECT now() AT TIME ZONE 'Asia/Kolkata' AS ist_now;")
    display(df_now)
    print("✅ DB connected and responding")
except Exception as e:
    print("❌ DB connectivity issue:", repr(e))
    print("Check SSH tunnel + PG* environment variables before running backtests.")

from datetime import date
from pathlib import Path

from backtest_engine import run_batch
from backtest_engine.postgres_provider import PostgresDataProvider
from backtest_engine.config import (
    BacktestConfig,
    StudyWindow,
    UniverseConfig,
    SignalConfig,
    SelectionConfig,
    RebalanceConfig,
    FeeConfig,
)

provider = PostgresDataProvider()

base_window = StudyWindow(start=date(2014, 1, 1), end=date(2024, 1, 1))

configs = [
    BacktestConfig(
        name="mf-10y-annual-top15",
        study_window=base_window,
        universe=UniverseConfig(preset="equity_active_direct"),
        signal=SignalConfig(
            name="rank_12m_category",
            direction="asc",
        ),
        selection=SelectionConfig(
            mode="top_n",
            top_n=15,
            min_funds=10,
            weight_scheme="equal",
        ),
        rebalance=RebalanceConfig(frequency="12M"),
        fees=FeeConfig(apply=True, annual_bps=100.0),
    ),
    BacktestConfig(
        name="mf-10y-annual-top30",
        study_window=base_window,
        universe=UniverseConfig(preset="equity_active_direct"),
        signal=SignalConfig(
            name="rank_12m_category",
            direction="asc",
        ),
        selection=SelectionConfig(
            mode="top_n",
            top_n=30,
            min_funds=20,
            weight_scheme="equal",
        ),
        rebalance=RebalanceConfig(frequency="12M"),
        fees=FeeConfig(apply=True, annual_bps=100.0),
    ),
]

batch_out_dir = Path("batch_outputs")

batch_summary = run_batch(
    configs=configs,
    data_provider=provider,
    out_dir=batch_out_dir,
    level="standard",  # or "light" / "full"
)

batch_summary


# Period-level view: one row per rebalance period

display(
    result.portfolio_periods[
        [
            "period_no",
            "start_date",
            "end_date",
            "period_days",
            "num_funds",
            "gross_return",
            "net_return",
            "benchmark_return",
            "alpha_return",
            "net_alpha_return",
        ]
    ]
    .head(10)
)

# Holdings view: first few rows of fund-level detail
display(
    result.holdings[
        [
            "run_id",
            "period_no",
            "rebalance_date",
            "schemecode",
            "scheme_name",
            "weight",
            "fund_return",
            "period_gross_return",
            "period_net_return",
        ]
    ]
    .head(20)
)
```

## db.py

```python
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Global cached engine so we don't recreate it on every query
_engine: Optional[Engine] = None


def get_engine() -> Engine:
    """Create (or return cached) SQLAlchemy engine for the Kairo Postgres DB.

    This assumes you've already started the SSH tunnel in your terminal:
        ./scripts/tunnel.sh start

    And that PGPASSWORD is exported in your shell.
    """
    global _engine
    if _engine is not None:
        return _engine

    PGHOST = os.getenv("PGHOST", "localhost")
    PGPORT = os.getenv("PGPORT", "5433")
    PGDATABASE = os.getenv("PGDATABASE", "kairo_production")
    PGUSER = os.getenv("PGUSER", "sanjay_readonly")
    PGPASSWORD = os.getenv("PGPASSWORD")

    if not PGPASSWORD:
        raise RuntimeError(
            "PGPASSWORD not found. Export it in your terminal before running backtests."
        )

    url = (
        f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}"
        f"@{PGHOST}:{PGPORT}/{PGDATABASE}?sslmode=require"
    )

    _engine = create_engine(url)
    return _engine


def run_sql(query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Run a SQL query against the Postgres DB and return a pandas DataFrame."""
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)

```

## config.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Literal, Optional


RebalanceFreq = Literal["NONE", "3M", "6M", "12M", "18M", "24M"]


@dataclass
class StudyWindow:
    start: Optional[date] = None   # None = auto_min
    end: Optional[date] = None     # None = auto_max


@dataclass
class UniverseConfig:
    # Either use a preset name OR raw filters (we'll implement presets later)
    preset: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SignalConfig:
    """Configuration for how to compute the ranking score.

    Two modes:

    1) Simple column mode (current behaviour):
       - Use `name`, `source`, `lookback_months`, `rank_scope`, etc.
       - Leave `expression` and `filter_expression` as None.
       - Engine will resolve a single column via _resolve_signal_column()
         and use that as `score`.

    2) Expression mode:
       - Set `expression` to a formula using columns in performance_ranking,
         e.g. "0.5 * perf_1y + 0.5 * perf_3y".
       - Optionally set `filter_expression` to a boolean formula to filter funds,
         e.g. "aum_cr > 300 and age_years > 3".
       - `name` becomes just a logical label for this signal.

    direction:
       - "asc": lower score is better (e.g. ranks).
       - "desc": higher score is better (e.g. returns).
    """
    name: str                       # e.g. "rank_12m_category"
    source: str = "performance_ranking"
    lookback_months: int = 12
    direction: Literal["asc", "desc"] = "desc"
    rank_scope: Literal["category", "asset_class", "universe"] = "category"
    tie_breaker: Optional[str] = None

    # Optional score expression, e.g. "0.5 * perf_1y + 0.5 * perf_3y"
    expression: Optional[str] = None

    # Optional filter expression, e.g. "aum_cr > 300 and age_years > 3"
    filter_expression: Optional[str] = None


@dataclass
class SelectionConfig:
    """Controls how many funds are selected into the portfolio.

    mode:
      - "top_n": pick the top `top_n` funds by score (current behaviour).
      - "all":   pick *all* eligible funds (ignores `top_n`, but still enforces `min_funds`).

    top_n:
      - used only when mode == "top_n".

    min_funds:
      - if the number of eligible funds is less than this, the engine will raise.

    weight_scheme:
      - currently only "equal" is supported: equal-weight across selected funds.
    """
    mode: Literal["top_n", "all"] = "top_n"
    top_n: int = 15
    min_funds: int = 10
    weight_scheme: Literal["equal"] = "equal"


@dataclass
class RebalanceConfig:
    frequency: RebalanceFreq = "12M"
    anchor: Literal["month_end"] = "month_end"


@dataclass
class FeeConfig:
    apply: bool = False
    annual_bps: float = 0.0   # 50 = 0.50% p.a.
    apply_frequency: Literal["daily", "monthly", "annual"] = "daily"


@dataclass
class TaxConfig:
    apply: bool = False
    stcg_rate: float = 0.15
    ltcg_rate: float = 0.10
    ltcg_holding_days: int = 365


@dataclass
class BacktestConfig:
    """Top-level config object for a single backtest run.

    This should be directly mappable from your Notion Variant row.
    """

    name: str
    study_window: StudyWindow
    universe: UniverseConfig
    signal: SignalConfig
    selection: SelectionConfig
    rebalance: RebalanceConfig
    fees: FeeConfig = field(default_factory=FeeConfig)
    tax: TaxConfig = field(default_factory=TaxConfig)

    # free-form metadata (links to Notion, code, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)

```

## formula.py

```python
# src/backtest_engine/formula.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable

import ast

import numpy as np
import pandas as pd
from sqlalchemy import inspect
from sqlalchemy.engine import Engine


# ---------- Metadata: which fields are allowed in formulas ----------


@dataclass
class FieldInfo:
    table: str
    name: str
    type: str  # textual DB type, e.g. "numeric", "double precision", etc.


def load_table_columns(
    engine: Engine,
    table_name: str,
    schema: str = "public",
) -> Dict[str, FieldInfo]:
    """
    Introspect a single table/view and return a mapping of column_name -> FieldInfo.
    """
    inspector = inspect(engine)
    columns = inspector.get_columns(table_name, schema=schema)

    fields: Dict[str, FieldInfo] = {}
    for col in columns:
        col_name = col["name"]
        col_type = str(col["type"])
        fields[col_name] = FieldInfo(
            table=table_name,
            name=col_name,
            type=col_type,
        )
    return fields


def load_selection_field_registry(
    engine: Engine,
    schema: str = "public",
) -> Dict[str, FieldInfo]:
    """
    Load the registry of *allowed* fields for SELECTION formulas.

    For now, we restrict this to the `performance_ranking` table only.
    Later you can extend this to a view or additional tables.
    """
    TABLE = "performance_ranking"
    return load_table_columns(engine, TABLE, schema=schema)


# ---------- Safe-ish expression evaluator over a DataFrame ----------


def _zscore(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / x.std(ddof=0)


SAFE_FUNCS: Dict[str, Callable[..., Any]] = {
    "abs": np.abs,
    "log": np.log,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "zscore": _zscore,
    "min": np.minimum,
    "max": np.maximum,
    "pow": np.power,  # allow pow(a, b) for exponent
    # add more if/when needed
}


ALLOWED_AST_NODES: Iterable[type] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num,
    ast.Name,
    ast.Load,
    ast.Call,
    ast.Compare,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.Eq,
    ast.NotEq,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.USub,
)


class FormulaSyntaxError(ValueError):
    """Raised when the formula string has invalid or forbidden syntax."""


class FormulaNameError(ValueError):
    """Raised when the formula references an unknown column or function."""


def _validate_ast(tree: ast.AST, allowed_names: Dict[str, Any]) -> None:
    """
    Walk the parsed AST and ensure:
    - only whitelisted node types are used
    - only whitelisted functions are used
    - all variable names exist in allowed_names
    """
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST_NODES):
            raise FormulaSyntaxError(f"Forbidden expression element: {type(node).__name__}")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise FormulaSyntaxError("Only simple function calls are allowed")
            if node.func.id not in SAFE_FUNCS:
                raise FormulaSyntaxError(f"Function '{node.func.id}' is not allowed")

        if isinstance(node, ast.Name):
            if node.id not in allowed_names and node.id not in SAFE_FUNCS:
                raise FormulaNameError(f"Unknown variable or function: '{node.id}'")


def evaluate_formula_on_df(
    df: pd.DataFrame,
    formula: str,
    allowed_fields: Dict[str, FieldInfo],
) -> pd.Series:
    """
    Evaluate a formula string against a DataFrame.

    - `df` must contain the columns referenced in the formula.
    - `allowed_fields` defines which column names are permitted in formulas;
      for now these are the columns of `performance_ranking`.
    - Returns a Pandas Series (numeric or boolean), indexed like df.

    Supports:
    - arithmetic: +, -, *, /, %, ^, **
    - comparisons: >, >=, <, <=, ==, !=
    - boolean: and, or (also & and | via numpy/pandas)
    - functions: those in SAFE_FUNCS (abs, log, zscore, pow, ...)
    """

    formula = formula.strip()
    if not formula:
        raise ValueError("Formula is empty")

    # Treat ^ as exponent (power), not bitwise XOR.
    # Users can write perf_6m ^ 2 and it becomes perf_6m ** 2.
    formula = formula.replace("^", "**")

    # Build the evaluation context: column_name -> Series
    context: Dict[str, Any] = {}

    for col in df.columns:
        # Only expose columns that are in allowed_fields
        if col in allowed_fields:
            context[col] = df[col]

    # Add allowed functions
    context.update(SAFE_FUNCS)

    # Parse & validate
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as e:
        raise FormulaSyntaxError(f"Invalid formula syntax: {e}") from e

    _validate_ast(tree, context)

    # Compile expression
    compiled = compile(tree, filename="<formula>", mode="eval")

    # Evaluate; Pandas/NumPy will broadcast operations across Series
    result = eval(compiled, {"__builtins__": {}}, context)

    if not isinstance(result, (pd.Series, np.ndarray)):
        # If it’s a scalar, broadcast to a Series
        result = pd.Series(result, index=df.index)

    # Coerce numpy array to Series if needed
    if isinstance(result, np.ndarray):
        result = pd.Series(result, index=df.index)

    return result
```

## postgres_provider.py

```python
from __future__ import annotations

from datetime import date
from typing import Iterable, List

import os
import pandas as pd

from .data_provider import DataProvider
from .config import UniverseConfig, SignalConfig
from .db import run_sql, get_engine

_BENCHMARK_CACHE: dict[str, pd.DataFrame] = {}

from .formula import (
    evaluate_formula_on_df,
    load_selection_field_registry,
    FormulaSyntaxError,
    FormulaNameError,
)

def _load_nifty500_tri() -> pd.DataFrame:
    """Load Nifty 500 TRI levels from CSV and cache in memory.

    Expected CSV format:
        date,value

    - `date` parseable as YYYY-MM-DD
    - `value` numeric index level
    """
    global _BENCHMARK_CACHE
    if "NIFTY_500_TRI" in _BENCHMARK_CACHE:
        return _BENCHMARK_CACHE["NIFTY_500_TRI"]

    # Allow overriding via env var; else use project-relative default
_BENCHMARK_CACHE: dict[str, pd.DataFrame] = {}


def _load_nifty500_tri() -> pd.DataFrame:
    """Load Nifty 500 TRI levels from CSV and cache in memory.

    Expected CSV location (by default):
        src/backtest_engine/data/market/nifty500_tri.csv

    Expected CSV format:
        date,value
    """
    global _BENCHMARK_CACHE
    if "NIFTY_500_TRI" in _BENCHMARK_CACHE:
        return _BENCHMARK_CACHE["NIFTY_500_TRI"]

    # 1) Allow override via env var if you ever want
    csv_path = os.getenv("NIFTY500_TRI_CSV")
    if not csv_path:
        # 2) Default: src/backtest_engine/data/market/nifty500_tri.csv
        here = os.path.dirname(__file__)  # .../src/backtest_engine
        csv_path = os.path.join(here, "data", "market", "nifty500_tri.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Nifty 500 TRI CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)

    _BENCHMARK_CACHE["NIFTY_500_TRI"] = df
    return df

    df = pd.read_csv(csv_path)
    # Expect columns 'date' and 'value'
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)

    _BENCHMARK_CACHE["NIFTY_500_TRI"] = df
    return df


class PostgresDataProvider(DataProvider):
    """DataProvider implementation backed by your Kairo Postgres DB."""

    # ---------- Universe -------------------------------------------------

    def get_universe(
        self,
        as_of: date,
        universe_config: UniverseConfig,
    ) -> pd.DataFrame:
        """Return investible universe on the given date.

        For now we implement a single preset:
          preset="equity_active_direct"

        Later we'll use universe_config.filters to make this more flexible
        (AUM filters, category excludes, etc.).
        """
        preset = universe_config.preset

        if preset != "equity_active_direct":
            raise NotImplementedError(
                f"Universe preset '{preset}' not implemented yet. "
                "Use preset='equity_active_direct' for now."
            )

        # NOTE: We are *not yet* using `as_of` for time-varying availability.
        # This is okay as a first pass and we can tighten later if needed.
        query = """
        SELECT
            sd.schemecode,
            sd.s_name AS scheme_name,
            sd.classcode,
            sc.category,
            sc.asset_type,
            pm.plan AS plan_name
        FROM scheme_details sd
        JOIN sclass_mst sc
          ON sd.classcode = sc.classcode
        JOIN plan_mst pm
          ON sd.plan = pm.plan_code
        WHERE
            sc.asset_type = 'Equity'          -- equity only
            AND sd.status ILIKE 'Active%%'    -- active only
            AND pm.plan_code = 5              -- direct plans
            AND sd.IsPurchaseAvailable = 'Y'  -- investible
            AND sd.Liquidated_Date IS NULL    -- not liquidated
            AND sd.dividendoptionflag = 'Z'   -- growth option
        ;
        """

        df = run_sql(query)
        return df

    # ---------- Signals --------------------------------------------------

    def _resolve_signal_column(self, signal_config: SignalConfig) -> str:
        """Map SignalConfig to a performance_ranking column name.

        For now we hard-code a simple mapping.
        """
        # Example: our "rank_12m_category" will use rank_1y_category column.
        if signal_config.name == "rank_12m_category":
            return "rank_1y_category"
        elif signal_config.name == "rank_12m_asset_class":
            return "rank_1y_asset_class"

        # You can extend this mapping as you add more signal names.
        raise NotImplementedError(
            f"Signal '{signal_config.name}' not mapped to a column yet."
        )

    def get_signal_scores(
        self,
        as_of: date,
        schemecodes: Iterable[int],
        signal_config: SignalConfig,
    ) -> pd.DataFrame:
        """Return signal scores as of `as_of` (no lookahead).

        Logic:
          - Use performance_ranking for Direct plans (plan_code = 5).
          - For each schemecode, pick the latest row with `date <= as_of`.
          - Optionally apply filter_expression to remove some funds.
          - Compute 'score' either from an expression or a single column.
        """
        # 1) Load *full* snapshot from performance_ranking (latest <= as_of, direct plan)
        query = """
        WITH latest AS (
            SELECT
                pr.*,
                ROW_NUMBER() OVER (
                    PARTITION BY pr.schemecode
                    ORDER BY pr.date DESC
                ) AS rn
            FROM performance_ranking pr
            WHERE
                pr.plan_code = 5
                AND pr.date <= :as_of
        )
        SELECT *
        FROM latest
        WHERE rn = 1;
        """
        df = run_sql(query, params={"as_of": as_of})

        if df.empty:
            raise RuntimeError(
                f"No performance_ranking data found for as_of={as_of}. "
                "Try using a later study_window.start, or check the available "
                "date range in performance_ranking for plan_code=5."
            )

        # 2) Optionally restrict to schemecodes passed in
        if schemecodes:
            codes_list = list(schemecodes)
            df = df[df["schemecode"].isin(codes_list)]

        if df.empty:
            raise RuntimeError(
                f"No performance_ranking rows found for given schemecodes as_of={as_of}."
            )

        # Drop the window helper column if present
        if "rn" in df.columns:
            df = df.drop(columns=["rn"])

        # 2.5) Load allowed field metadata from performance_ranking
        #      (this enforces that formulas only use valid columns)
        engine = get_engine()
        field_registry = load_selection_field_registry(engine)

        # 3) Apply optional filter_expression (on the snapshot)
        if signal_config.filter_expression:
            expr = signal_config.filter_expression
            try:
                mask = evaluate_formula_on_df(
                    df=df,
                    formula=expr,
                    allowed_fields=field_registry,
                )
            except (FormulaSyntaxError, FormulaNameError, Exception) as exc:
                raise ValueError(
                    f"Error evaluating filter_expression={expr!r} "
                    f"on performance_ranking snapshot"
                ) from exc

            # Ensure boolean; treat NaNs as False
            mask = mask.astype(bool).fillna(False)
            df = df[mask]

            if df.empty:
                raise RuntimeError(
                    f"filter_expression removed all funds as_of={as_of}: {expr!r}"
                )

        # 4) Compute 'score' either from expression or from a resolved column
        if signal_config.expression:
            expr = signal_config.expression
            try:
                scores = evaluate_formula_on_df(
                    df=df,
                    formula=expr,
                    allowed_fields=field_registry,
                )
            except (FormulaSyntaxError, FormulaNameError, Exception) as exc:
                raise ValueError(
                    f"Error evaluating signal expression={expr!r} "
                    f"on performance_ranking snapshot"
                ) from exc

            df = df.copy()
            df["score"] = scores
        else:
            # Fallback to simple column mode using existing mapping logic
            col = self._resolve_signal_column(signal_config)
            if col not in df.columns:
                raise KeyError(
                    f"Resolved signal column {col!r} not in performance_ranking "
                    f"columns: {list(df.columns)}"
                )
            df = df.copy()
            df["score"] = df[col]

        # 5) Apply direction: "asc" = lower is better, "desc" = higher is better
        if signal_config.direction == "asc":
            df["score"] = -df["score"]

        # Final output: schemecode + score
        return df[["schemecode", "score"]]

    # ---------- NAV series ----------------------------------------------

    def get_nav_series(
        self,
        schemecodes: Iterable[int],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Return NAV history for all schemecodes over [start, end].

        Expected columns in output:
            ['date', 'schemecode', 'nav']
        """
        codes = list({int(x) for x in schemecodes})
        if not codes:
            return pd.DataFrame(columns=["date", "schemecode", "nav"])

        query = """
        SELECT
            schemecode,
            navdate::date AS date,
            navrs       AS nav
        FROM navhist
        WHERE
            schemecode = ANY(:schemecodes)
            AND navdate BETWEEN :start AND :end
        ORDER BY date, schemecode
        ;
        """

        params = {
            "schemecodes": codes,
            "start": start,
            "end": end,
        }

        df = run_sql(query, params=params)
        return df

    # ---------- Benchmark series ----------------------------------------
    def get_benchmark_series(
        self,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Return benchmark (Nifty 500 TRI) index levels over [start, end].

        Uses a CSV file loaded via _load_nifty500_tri().
        """
        if start >= end:
            raise ValueError(f"start ({start}) must be before end ({end})")

        bench = _load_nifty500_tri()
        mask = (bench["date"] >= start) & (bench["date"] <= end)
        window = bench.loc[mask].copy()

        if window.empty:
            raise RuntimeError(
                f"No Nifty 500 TRI data between {start} and {end}. "
                "Check your CSV coverage."
            )

        return window
```

## batch.py

```python
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Literal, Union

import pandas as pd

from .config import BacktestConfig
from .data_provider import DataProvider
from .engine import BacktestEngine, BacktestResult

SaveLevel = Literal["light", "standard", "full"]

def run_batch(
    configs: Iterable[BacktestConfig],
    data_provider: DataProvider,
    out_dir: Union[str, Path],
    level: SaveLevel = "standard",
) -> pd.DataFrame:
    """Run a batch of backtests and save outputs for each run.

    Parameters
    ----------
    configs : Iterable[BacktestConfig]
        A sequence/list of BacktestConfig objects to run.
    data_provider : DataProvider
        The data provider (e.g. PostgresDataProvider) to use for all runs.
    out_dir : str or Path
        Root directory where outputs for this batch will be written.
        Each run gets its own subdirectory under this.
    level : {"light", "standard", "full"}, default "standard"
        Passed through to BacktestResult.save():
        - "light":    summary only
        - "standard": summary + portfolio_periods
        - "full":     summary + portfolio_periods + holdings (+ config.yaml)

    Returns
    -------
    pd.DataFrame
        Concatenated summary table for all runs, with extra columns
        indicating where outputs were written.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    engine = BacktestEngine(data_provider)

    summary_frames = []

    for cfg in configs:
        run_id = cfg.name or "backtest"
        run_dir = out_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Run the backtest
        result: BacktestResult = engine.run(cfg)

        # Save outputs for this run
        paths: Dict[str, Path] = result.save(run_dir, level=level)

        # Enrich summary with batch metadata
        summary = result.summary.copy()
        summary["output_dir"] = str(run_dir)
        summary["output_files"] = ", ".join(str(p) for p in paths.values())

        summary_frames.append(summary)

    if not summary_frames:
        return pd.DataFrame()

    batch_summary = pd.concat(summary_frames, ignore_index=True)

    # Also save a batch-level summary CSV at the root
    batch_summary_path = out_root / "batch_summary.csv"
    batch_summary.to_csv(batch_summary_path, index=False)

    return batch_summary

```

## __init__.py

```python
from .engine import BacktestEngine
from .postgres_provider import PostgresDataProvider
from .batch import run_batch

__all__ = [
    "BacktestEngine",
    "PostgresDataProvider",
    "run_batch",
]
```

## engine.py

```python
from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import pandas as pd

from .config import BacktestConfig
from .data_provider import DataProvider
from .utils.dates import generate_rebalance_dates

class BacktestResult:
    """Container for engine outputs and helper methods.

    Attributes
    ----------
    run_id : str
        Identifier for this run (typically config.name).
    summary : pd.DataFrame
        One row per run, with total returns, CAGR, alpha, etc.
    portfolio_periods : pd.DataFrame
        One row per rebalance period with period-level returns.
    holdings : pd.DataFrame
        One row per fund per period with weights and period returns.
    config : Optional[BacktestConfig]
        Optional copy of the BacktestConfig used to run this backtest.
    """

    def __init__(
        self,
        run_id: str,
        summary: pd.DataFrame,
        portfolio_periods: pd.DataFrame,
        holdings: pd.DataFrame,
        config: Optional[BacktestConfig] = None,
    ):
        self.run_id = run_id
        self.summary = summary
        self.portfolio_periods = portfolio_periods
        self.holdings = holdings
        self.config = config

    def save(
        self,
        out_dir: Union[str, Path],
        level: Literal["light", "standard", "full"] = "standard",
    ) -> Dict[str, Path]:
        """Save result tables to CSV.

        Parameters
        ----------
        out_dir : str or Path
            Directory where CSVs will be written. It will be created if needed.
        level : {"light", "standard", "full"}, default "standard"
            - "light":    summary only
            - "standard": summary + portfolio_periods
            - "full":     summary + portfolio_periods + holdings (+ config if available)

        Returns
        -------
        Dict[str, Path]
            Mapping of logical name -> file path written.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        prefix = self.run_id or "backtest"
        paths: Dict[str, Path] = {}

        # Always save summary
        summary_path = out_dir / f"{prefix}.summary.csv"
        self.summary.to_csv(summary_path, index=False)
        paths["summary"] = summary_path

        # Period-level table for standard/full
        if level in ("standard", "full"):
            periods_path = out_dir / f"{prefix}.periods.csv"
            self.portfolio_periods.to_csv(periods_path, index=False)
            paths["portfolio_periods"] = periods_path

        # Holdings only for full (can be large)
        if level == "full" and not self.holdings.empty:
            holdings_path = out_dir / f"{prefix}.holdings.csv"
            self.holdings.to_csv(holdings_path, index=False)
            paths["holdings"] = holdings_path

        # Optional: dump config as YAML for full runs, if available
        if level == "full" and self.config is not None:
            try:
                import yaml  # PyYAML is already in requirements.txt

                config_path = out_dir / f"{prefix}.config.yaml"
                with config_path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(asdict(self.config), f, sort_keys=False)
                paths["config"] = config_path
            except Exception:
                # If PyYAML is missing or anything goes wrong, just skip config dump
                pass

        return paths

class BacktestEngine:
    """Core engine orchestrator.

    Current capabilities:
      - Single portfolio over a study window, with optional rebalancing.
      - Universe via DataProvider.get_universe().
      - Signals via DataProvider.get_signal_scores() (no lookahead).
      - SelectionConfig: mode="top_n" or "all", equal-weight.
      - NAV walking via DataProvider.get_nav_series().
      - Benchmark (Nifty 500 TRI) via DataProvider.get_benchmark_series().
      - Optional fee drag via BacktestConfig.fees (no tax yet).
    """

    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider

    def run(self, config: BacktestConfig) -> BacktestResult:
        """Run a backtest for a single portfolio, possibly with rebalancing."""
        run_id = config.name

        if config.study_window.start is None or config.study_window.end is None:
            raise ValueError("study_window.start and end must be set")

        start_date: date = config.study_window.start
        end_date: date = config.study_window.end

        # Generate all rebalance dates inside [start, end)
        rebalance_dates = generate_rebalance_dates(
            start=start_date,
            end=end_date,
            frequency=config.rebalance.frequency,
        )

        if not rebalance_dates:
            raise RuntimeError("No rebalance dates generated for this config.")

        period_rows: List[Dict] = []
        holdings_frames: List[pd.DataFrame] = []

        # Equity curves
        equity_gross = 1.0   # compound portfolio returns (before fees)
        equity_net = 1.0     # compound portfolio returns (after fees)
        equity_bench = 1.0   # compound benchmark returns

        # Fee setup
        fee_enabled = bool(
            getattr(config, "fees", None)
            and config.fees.apply
            and config.fees.annual_bps > 0
        )
        fee_annual_rate = (config.fees.annual_bps / 10_000.0) if fee_enabled else 0.0

        for idx, rebalance_date in enumerate(rebalance_dates):
            period_no = idx + 1
            period_start = rebalance_date
            period_end = (
                rebalance_dates[idx + 1]
                if idx + 1 < len(rebalance_dates)
                else end_date
            )
            period_days = (period_end - period_start).days

            # 1) Get investible universe on the rebalance date
            universe_df = self.data_provider.get_universe(
                rebalance_date, config.universe
            )
            if universe_df.empty:
                raise RuntimeError(f"No eligible universe on {rebalance_date}")

            # 2) Get signal scores and select funds
            scores_df = self.data_provider.get_signal_scores(
                as_of=rebalance_date,
                schemecodes=universe_df["schemecode"].tolist(),
                signal_config=config.signal,
            )

            merged = (
                universe_df.merge(scores_df, on="schemecode", how="inner")
                .sort_values("score", ascending=(config.signal.direction == "asc"))
            )

            # Selection modes
            if config.selection.mode == "top_n":
                top = merged.head(config.selection.top_n).copy()
            elif config.selection.mode == "all":
                top = merged.copy()
            else:
                raise ValueError(
                    f"Unsupported selection.mode={config.selection.mode!r}"
                )

            n = len(top)

            if n < config.selection.min_funds:
                raise RuntimeError(
                    f"Only {n} eligible funds on {rebalance_date}, "
                    f"min_funds={config.selection.min_funds}"
                )

            # 3) We currently only support equal-weight portfolios
            if config.selection.weight_scheme != "equal":
                raise ValueError(
                    f"Unsupported weight_scheme={config.selection.weight_scheme!r}"
                )
            top["weight"] = 1.0 / n

            # 4) Pull NAV history for this period
            nav_df = self.data_provider.get_nav_series(
                schemecodes=top["schemecode"].tolist(),
                start=period_start,
                end=period_end,
            )

            if nav_df.empty:
                raise RuntimeError(
                    "No NAV data returned for selected schemes between "
                    f"{period_start} and {period_end}"
                )

            # 5) Compute fund-level returns within this period
            nav_sorted = nav_df.sort_values(["schemecode", "date"])
            first_nav = (
                nav_sorted.groupby("schemecode").first()["nav"].rename("nav_start")
            )
            last_nav = (
                nav_sorted.groupby("schemecode").last()["nav"].rename("nav_end")
            )

            nav_stats = pd.concat([first_nav, last_nav], axis=1).reset_index()
            nav_stats["fund_return"] = (
                nav_stats["nav_end"] / nav_stats["nav_start"] - 1.0
            )

            # Merge returns into holdings for this period
            top = top.merge(nav_stats, on="schemecode", how="left")

            # 6) Portfolio-level gross return and CAGR for this period
            top["contribution"] = top["weight"] * top["fund_return"]
            gross_return = top["contribution"].sum()

            if period_days > 0:
                gross_cagr = (1.0 + gross_return) ** (365.0 / period_days) - 1.0
            else:
                gross_cagr = float("nan")

            # 7) Benchmark return & CAGR for this period
            bench_df = self.data_provider.get_benchmark_series(
                start=period_start,
                end=period_end,
            )
            bench_sorted = bench_df.sort_values("date")
            bench_start = bench_sorted["value"].iloc[0]
            bench_end = bench_sorted["value"].iloc[-1]
            bench_return = bench_end / bench_start - 1.0

            if period_days > 0:
                bench_cagr = (1.0 + bench_return) ** (365.0 / period_days) - 1.0
            else:
                bench_cagr = float("nan")

            # 8) Fee drag & net returns for this period
            if fee_enabled and period_days > 0:
                # Continuous-ish fee accrual from annual rate
                fee_factor_period = (1.0 - fee_annual_rate) ** (
                    period_days / 365.0
                )
                fee_return = fee_factor_period - 1.0  # negative
            else:
                fee_factor_period = 1.0
                fee_return = 0.0

            net_return = (1.0 + gross_return) * fee_factor_period - 1.0

            if period_days > 0:
                net_cagr = (1.0 + net_return) ** (365.0 / period_days) - 1.0
            else:
                net_cagr = float("nan")

            # 9) Update compounded equity curves
            equity_gross *= (1.0 + gross_return)
            equity_net *= (1.0 + net_return)
            equity_bench *= (1.0 + bench_return)

            # Store period-level row
            period_rows.append(
                {
                    "run_id": run_id,
                    "period_no": period_no,
                    "rebalance_date": rebalance_date,
                    "start_date": period_start,
                    "end_date": period_end,
                    "period_days": period_days,
                    "num_funds": n,
                    "gross_return": gross_return,
                    "gross_cagr": gross_cagr,
                    "net_return": net_return,
                    "net_cagr": net_cagr,
                    "fee_return": fee_return,
                    "benchmark_return": bench_return,
                    "benchmark_cagr": bench_cagr,
                    "alpha_return": gross_return - bench_return,
                    "alpha_cagr": gross_cagr - bench_cagr,
                    "net_alpha_return": net_return - bench_return,
                    "net_alpha_cagr": net_cagr - bench_cagr,
                }
            )

            # Store holdings for this period
            holdings_frames.append(
                top.assign(
                    run_id=run_id,
                    period_no=period_no,
                    rebalance_date=rebalance_date,
                    period_start=period_start,
                    period_end=period_end,
                    period_days=period_days,
                    period_gross_return=gross_return,
                    period_gross_cagr=gross_cagr,
                    period_net_return=net_return,
                    period_net_cagr=net_cagr,
                    period_fee_return=fee_return,
                    period_benchmark_return=bench_return,
                    period_benchmark_cagr=bench_cagr,
                    period_alpha_return=gross_return - bench_return,
                    period_alpha_cagr=gross_cagr - bench_cagr,
                    period_net_alpha_return=net_return - bench_return,
                    period_net_alpha_cagr=net_cagr - bench_cagr,
                )
            )

        if not period_rows:
            raise RuntimeError("No valid periods generated for this run.")

        # 10) Backtest-level summary (compounded over all periods)
        total_days = (end_date - start_date).days
        gross_return_total = equity_gross - 1.0
        net_return_total = equity_net - 1.0
        bench_return_total = equity_bench - 1.0

        if total_days > 0:
            gross_cagr_total = equity_gross ** (365.0 / total_days) - 1.0
            net_cagr_total = equity_net ** (365.0 / total_days) - 1.0
            bench_cagr_total = equity_bench ** (365.0 / total_days) - 1.0
        else:
            gross_cagr_total = float("nan")
            net_cagr_total = float("nan")
            bench_cagr_total = float("nan")

        summary = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "name": config.name,
                    "start_date": start_date,
                    "end_date": end_date,
                    "num_periods": len(period_rows),
                    "gross_return": gross_return_total,
                    "gross_cagr": gross_cagr_total,
                    "net_return": net_return_total,
                    "net_cagr": net_cagr_total,
                    "benchmark_return": bench_return_total,
                    "benchmark_cagr": bench_cagr_total,
                    "alpha_return": gross_return_total - bench_return_total,
                    "alpha_cagr": gross_cagr_total - bench_cagr_total,
                    "net_alpha_return": net_return_total - bench_return_total,
                    "net_alpha_cagr": net_cagr_total - bench_cagr_total,
                    "fees_applied": fee_enabled,
                    "fees_annual_bps": (
                        config.fees.annual_bps if fee_enabled else 0.0
                    ),
                }
            ]
        )

        portfolio_periods = pd.DataFrame(period_rows)
        holdings = (
            pd.concat(holdings_frames, ignore_index=True)
            if holdings_frames
            else pd.DataFrame()
        )

        return BacktestResult(
            run_id=run_id,
            summary=summary,
            portfolio_periods=portfolio_periods,
            holdings=holdings,
            config=config,
        )
```

## data_provider.py

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Iterable, Protocol

import pandas as pd


class DataProvider(ABC):
    """Abstract interface for all data access.

    We'll implement a PostgresDataProvider that talks to your DB via SQLAlchemy.
    """

    @abstractmethod
    def get_universe(
        self,
        as_of: date,
        universe_config,
    ) -> pd.DataFrame:
        """Return universe of schemes valid on as_of date.

        Must include at least: schemecode, scheme_name, classcode, category, etc.
        """

    @abstractmethod
    def get_signal_scores(
        self,
        as_of: date,
        schemecodes: Iterable[int],
        signal_config,
    ) -> pd.DataFrame:
        """Return signal scores for given schemecodes at as_of date.

        Output: columns ['schemecode', 'score'] (and anything else you want).
        """

    @abstractmethod
    def get_nav_series(
        self,
        schemecodes: Iterable[int],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Return NAV history for all schemecodes over [start, end].

        Expected columns: ['date', 'schemecode', 'nav']
        """

    @abstractmethod
    def get_benchmark_series(
        self,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Return benchmark (Nifty 500 TRI) index levels over [start, end]."""

```

## selection.py

```python
# src/backtest_engine/selection.py

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy.engine import Engine

from .formula import (
    evaluate_formula_on_df,
    load_selection_field_registry,
)


@dataclass
class FormulaTopNConfig:
    """
    Config for 'formula_topn' selection.

    Example:
        formula = "0.5 * perf_6m + 0.5 * perf_12m"
        top_n = 10
    """
    formula: str
    top_n: int


def select_topn_by_formula(
    engine: Engine,
    universe_df: pd.DataFrame,
    cfg: FormulaTopNConfig,
    schema: str = "public",
) -> pd.DataFrame:
    """
    Given:
      - a SQLAlchemy Engine,
      - a DataFrame representing the current universe at a given date
        (already joined with performance_ranking),
      - a FormulaTopNConfig containing the formula + N,

    returns:
      - a DataFrame of the selected Top-N schemes, with an extra 'score' column.

    Assumes:
      - universe_df has one row per scheme (schemecode, etc.)
      - universe_df already includes all performance_ranking columns
        you want to use in the formula.
    """
    if universe_df.empty:
        return universe_df

    # 1) Load allowed field metadata (performance_ranking columns)
    field_registry = load_selection_field_registry(engine, schema=schema)

    # 2) Evaluate formula to get a score per row
    score = evaluate_formula_on_df(
        df=universe_df,
        formula=cfg.formula,
        allowed_fields=field_registry,
    )

    # 3) Attach and sort
    df_with_score = universe_df.copy()
    df_with_score["score"] = score

    # Drop rows with NaN score to avoid surprises
    df_with_score = df_with_score.dropna(subset=["score"])

    # 4) Take Top-N by descending score
    selected = df_with_score.sort_values("score", ascending=False).head(cfg.top_n)

    return selected
```

## dates.py

```python
from __future__ import annotations

from datetime import date
from typing import List

from dateutil.relativedelta import relativedelta


def _months_for_frequency(frequency: str) -> int:
    """Map rebalance frequency code to number of months."""
    if frequency == "NONE":
        return 0
    mapping = {
        "3M": 3,
        "6M": 6,
        "12M": 12,
        "18M": 18,
        "24M": 24,
    }
    if frequency not in mapping:
        raise ValueError(f"Unsupported rebalance frequency: {frequency}")
    return mapping[frequency]


def generate_rebalance_dates(start: date, end: date, frequency: str) -> List[date]:
    """Generate rebalance dates between start and end (inclusive of start).

    Rules:
      - If frequency == "NONE": a single rebalance at `start` (buy & hold).
      - Else: rebalance at `start`, then every N months, stopping
        when the next date would be >= end.

    Examples:
      start=2020-01-01, end=2021-01-01, freq="12M" -> [2020-01-01]
      start=2020-01-01, end=2021-01-01, freq="6M"  -> [2020-01-01, 2020-07-01]
    """
    if start >= end:
        raise ValueError(f"start ({start}) must be before end ({end})")

    months = _months_for_frequency(frequency)
    # No rebalancing: single portfolio from start to end
    if months == 0:
        return [start]

    dates: List[date] = [start]
    current = start

    while True:
        current = current + relativedelta(months=months)
        if current >= end:
            break
        dates.append(current)

    return dates

```

