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

from backtest_engine.app_settings import (
    load_app_settings,
    save_app_settings,
    load_universe_presets,
    save_universe_presets,
)
from backtest_engine.db import run_sql

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
    fee_apply: bool,
    fee_bps: float,
    tax_apply: bool,
    stcg_rate: float,
    ltcg_rate: float,
    ltcg_holding_days: int,
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
            apply=fee_apply,
            annual_bps=fee_bps,
        ),
        tax=TaxConfig(
            apply=tax_apply,
            stcg_rate=stcg_rate,
            ltcg_rate=ltcg_rate,
            ltcg_holding_days=ltcg_holding_days,
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
    
    # Load app-level settings (fees, tax, universes)
    app_settings = load_app_settings()
    universe_presets = load_universe_presets()

    tabs = st.tabs(["Run backtest", "Saved backtests", "Settings & universes"])

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
                options=sorted(universe_presets.keys()),
                index=0,
                help="Presets are defined under the Settings & universes tab.",
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

        with col_run_left:
            st.subheader("Universe & dates")
            ...
            st.subheader("Rebalancing")
            ...

            st.subheader("Fees & Tax (from Settings)")
            st.markdown(
                f"- **Fees**: {'ON' if app_settings.fees.apply else 'OFF'} "
                f"({app_settings.fees.annual_bps:.1f} bps p.a.)\n"
                f"- **Tax**: {'ON' if app_settings.tax.apply else 'OFF'} "
                f"(STCG {app_settings.tax.stcg_rate*100:.1f}%, "
                f"LTCG {app_settings.tax.ltcg_rate*100:.1f}%, "
                f"holding ≥ {app_settings.tax.ltcg_holding_days} days)"
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
                        fee_apply=app_settings.fees.apply,
                        fee_bps=float(app_settings.fees.annual_bps),
                        tax_apply=app_settings.tax.apply,
                        stcg_rate=float(app_settings.tax.stcg_rate),
                        ltcg_rate=float(app_settings.tax.ltcg_rate),
                        ltcg_holding_days=int(app_settings.tax.ltcg_holding_days),
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

    # -------------------------------------------------------------------------
    # TAB 3: SETTINGS & UNIVERSES
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.header("Global settings & universe presets")

        # ----- Fees & tax settings -----
        st.subheader("Fees & Tax defaults")

        fee_apply = st.checkbox(
            "Apply annual fee?",
            value=app_settings.fees.apply,
            help="If checked, portfolios will include an annual fee drag.",
        )
        fee_bps = st.number_input(
            "Fee (bps per year)",
            min_value=0.0,
            max_value=500.0,
            value=float(app_settings.fees.annual_bps),
            step=5.0,
        )

        tax_apply = st.checkbox(
            "Apply tax?",
            value=app_settings.tax.apply,
            help="Plumbing only for now; engine tax logic still to be added.",
        )
        col_tax1, col_tax2, col_tax3 = st.columns(3)
        with col_tax1:
            stcg_pct = st.number_input(
                "STCG rate (%)",
                min_value=0.0,
                max_value=50.0,
                value=float(app_settings.tax.stcg_rate * 100),
                step=0.5,
            )
        with col_tax2:
            ltcg_pct = st.number_input(
                "LTCG rate (%)",
                min_value=0.0,
                max_value=50.0,
                value=float(app_settings.tax.ltcg_rate * 100),
                step=0.5,
            )
        with col_tax3:
            ltcg_days = st.number_input(
                "LTCG holding period (days)",
                min_value=1,
                max_value=3650,
                value=int(app_settings.tax.ltcg_holding_days),
                step=30,
            )

        if st.button("💾 Save fee & tax settings"):
            app_settings.fees.apply = fee_apply
            app_settings.fees.annual_bps = float(fee_bps)
            app_settings.tax.apply = tax_apply
            app_settings.tax.stcg_rate = float(stcg_pct) / 100.0
            app_settings.tax.ltcg_rate = float(ltcg_pct) / 100.0
            app_settings.tax.ltcg_holding_days = int(ltcg_days)
            save_app_settings(app_settings)
            st.success("Fee & tax settings saved to config/app_settings.yaml")

        st.markdown("---")

        # ----- Universe presets -----
        st.subheader("Universe presets")

        # Show existing presets
        preset_names = sorted(universe_presets.keys())
        if preset_names:
            selected_preset = st.selectbox(
                "Existing presets",
                options=preset_names,
                help="Select a preset to inspect or edit.",
            )
            if selected_preset:
                p = universe_presets[selected_preset]
                st.markdown(f"**Description:** {p.get('description', '')}")
                st.json(p)
        else:
            st.info("No universe presets found yet. Create one below.")

        st.markdown("### Create / edit a preset")

        new_name = st.text_input(
            "Preset name (slug)",
            value="",
            placeholder="e.g. equity_flexicap_quality",
            help="Used in configs as UniverseConfig.preset",
        )
        new_desc = st.text_area(
            "Description",
            value="",
            height=60,
        )

        # ---- Pull universe candidates from DB ----
        st.caption("Universe building blocks from sclass_mst")

        @st.cache_data(show_spinner=False)
        def load_class_universe() -> pd.DataFrame:
            q = """
            SELECT DISTINCT
                classcode,
                asset_type,
                category,
                sub_category,
                classname
            FROM sclass_mst
            ORDER BY asset_type, category, sub_category, classname;
            """
            return run_sql(q)

        class_df = load_class_universe()

        asset_types = sorted(class_df["asset_type"].dropna().unique().tolist())
        selected_asset_types = st.multiselect(
            "Asset types",
            options=asset_types,
            default=["Equity"] if "Equity" in asset_types else asset_types,
        )

        # Filter categories by selected asset types for easier UX
        filt_df = class_df[class_df["asset_type"].isin(selected_asset_types)]

        categories = sorted(filt_df["category"].dropna().unique().tolist())
        selected_categories = st.multiselect(
            "Include categories",
            options=categories,
            default=[],
            help="Leave empty to include all categories under the chosen asset types.",
        )

        excluded_categories = st.multiselect(
            "Exclude categories",
            options=categories,
            default=[],
            help="Optional: categories to explicitly exclude.",
        )

        col_flags1, col_flags2 = st.columns(2)
        with col_flags1:
            only_direct = st.checkbox("Only Direct plans", value=True)
            only_active = st.checkbox("Only Active schemes", value=True)
        with col_flags2:
            investible_only = st.checkbox("Only investible today", value=True)
            growth_only = st.checkbox("Growth option only", value=True)

        if st.button("💾 Save / update universe preset"):
            if not new_name.strip():
                st.error("Preset name is required.")
            else:
                universe_presets[new_name] = {
                    "description": new_desc.strip(),
                    "asset_types": selected_asset_types,
                    "include_categories": selected_categories,
                    "exclude_categories": excluded_categories,
                    "only_direct": only_direct,
                    "only_active": only_active,
                    "investible_only": investible_only,
                    "growth_only": growth_only,
                }
                save_universe_presets(universe_presets)
                st.success(
                    f"Preset '{new_name}' saved to config/universe_presets.yaml "
                    "(commit this file to version it)."
                )

if __name__ == "__main__":
    main()
