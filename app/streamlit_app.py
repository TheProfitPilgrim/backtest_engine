from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
import streamlit as st

from backtest_engine.config import (
    BacktestConfig,
    StudyWindow,
    UniverseConfig,
    SignalConfig,
    SelectionConfig,
    RebalanceConfig,
    FeeConfig,
    TaxConfig,
    CohortConfig,
)
from backtest_engine.engine import BacktestEngine
from backtest_engine.postgres_provider import PostgresDataProvider

from backtest_engine.app_settings import (
    AppSettings,
    FeeSettings,
    TaxSettings,
    load_app_settings,
    load_universe_presets,
    save_app_settings,
    save_universe_presets,
)

# -------------------------------------------------------------------
# Paths & constants
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "outputs" / "single_runs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Helper: descriptive text for what the backtest is answering
# -------------------------------------------------------------------

def describe_backtest(
    mode: Literal["single", "rolling_cohort"],
    start_date: date,
    end_date: date,
    universe_preset: str,
    entry_signal_name: str,
    entry_direction: str,
    entry_mode: str,
    entry_top_n: int | None,
    rebalance_freq: str,
    has_separate_rebalance: bool,
    rebalance_signal_name: str | None,
    rebalance_direction: str | None,
    rebalance_mode: str | None,
    rebalance_top_n: int | None,
    cohort_start_freq: str | None,
    cohort_horizon_years: float | None,
    fees_apply: bool,
    fees_annual_bps: float,
) -> str:
    """Return an English description of the research question implied by the UI choices."""
    dir_map = {
        "asc": "lower values are better (ranks)",
        "desc": "higher values are better (returns / scores)",
    }
    freq_map = {
        "NONE": "no rebalancing (pure buy-and-hold)",
        "1M": "monthly rebalancing",
        "3M": "quarterly rebalancing",
        "6M": "half-yearly rebalancing",
        "12M": "annual rebalancing",
        "18M": "18-month rebalancing",
        "24M": "two-yearly rebalancing",
    }
    cohort_map = {
        "1M": "every month",
        "3M": "every 3 months",
        "6M": "every 6 months",
        "12M": "every year",
    }

    # Entry selection phrase
    if entry_mode == "all":
        selection_phrase = "all eligible funds from the universe"
    else:
        selection_phrase = f"the top {entry_top_n} funds from the universe"

    base = (
        f"If you invested a lumpsum into a portfolio of {selection_phrase} "
        f"‘{universe_preset}’ on the chosen start dates, "
        f"selected using the signal **{entry_signal_name}** "
        f"where {dir_map.get(entry_direction, 'higher scores are better')}, "
    )

    # Rebalance description
    rebalance_desc = ""
    if rebalance_freq == "NONE":
        rebalance_desc = "and then simply bought and held that portfolio without any rebalancing"
    else:
        # Same or separate criteria?
        if not has_separate_rebalance:
            rebalance_desc = (
                f"and then rebalanced the portfolio using the **same criteria** "
                f"{freq_map.get(rebalance_freq, 'on a fixed schedule')}"
            )
        else:
            # separate rebalance criteria
            if rebalance_mode == "all":
                reb_sel_phrase = "all eligible funds at each rebalance"
            else:
                reb_sel_phrase = f"the top {rebalance_top_n} funds at each rebalance"

            rebalance_desc = (
                f"and then rebalanced the portfolio {freq_map.get(rebalance_freq, '')}, "
                f"selecting {reb_sel_phrase} based on **{rebalance_signal_name}** "
                f"where {dir_map.get(rebalance_direction or 'desc', 'higher scores are better')}"
            )

    # Mode-specific horizon description
    if mode == "single":
        horizon_desc = (
            f"over the full backtest window from {start_date} to {end_date} "
            f"(a single portfolio starting on {start_date})."
        )
    else:
        horizon_desc = (
            f"starting a new cohort {cohort_map.get(cohort_start_freq or '1M', 'every month')} "
            f"between {start_date} and {end_date}, "
            f"and holding each cohort for approximately {cohort_horizon_years:.1f} years."
        )

    # Fee description
    if fees_apply and fees_annual_bps > 0:
        fee_desc = (
            f" We also deduct an annual fee of {fees_annual_bps:.1f} bps "
            f"(~{fees_annual_bps/100.0:.2f}% p.a.) from portfolio returns."
        )
    else:
        fee_desc = " No additional fees are deducted in this backtest."

    question = (
        base
        + rebalance_desc
        + ", what would the portfolio have returned net of these rules compared to a "
        "buy-and-hold Nifty 500 TRI benchmark? "
        + horizon_desc
        + fee_desc
    )

    return question


# -------------------------------------------------------------------
# Wizard helpers
# -------------------------------------------------------------------
def init_session_state():
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 1

    # Load persisted app-level defaults (fees & tax)
    app_settings = load_app_settings()

    # Basic fields defaults
    st.session_state.setdefault("run_name", "mf-10y-annual-top15")
    st.session_state.setdefault("mode", "single")
    st.session_state.setdefault("study_start", date(2014, 1, 1))
    st.session_state.setdefault("study_end", date(2024, 1, 1))

    # Universe (wizard Step 2 will pull the options from YAML)
    st.session_state.setdefault("universe_preset", "equity_active_direct")

    # Entry signal & selection
    st.session_state.setdefault("entry_signal_name", "rank_12m_category")
    st.session_state.setdefault("entry_signal_direction", "asc")  # ranks: lower is better
    st.session_state.setdefault("entry_selection_mode", "top_n")
    st.session_state.setdefault("entry_top_n", 15)
    st.session_state.setdefault("entry_min_funds", 10)

    # Rebalance
    st.session_state.setdefault("rebalance_frequency", "12M")
    st.session_state.setdefault("use_separate_rebalance", False)
    st.session_state.setdefault("rebalance_signal_name", "rank_12m_category")
    st.session_state.setdefault("rebalance_signal_direction", "asc")
    st.session_state.setdefault("rebalance_selection_mode", "top_n")
    st.session_state.setdefault("reb_top_n", 15)
    st.session_state.setdefault("reb_min_funds", 10)

    # Cohorts
    st.session_state.setdefault("cohort_start_frequency", "1M")
    st.session_state.setdefault("cohort_horizon_years", 3.0)

    # Fees & tax – seed from app-level defaults defined in app_settings.yaml
    st.session_state.setdefault("fees_apply", app_settings.fees.apply)
    st.session_state.setdefault("fees_annual_bps", app_settings.fees.annual_bps)

    st.session_state.setdefault("tax_apply", app_settings.tax.apply)
    st.session_state.setdefault("tax_stcg_rate", app_settings.tax.stcg_rate)
    st.session_state.setdefault("tax_ltcg_rate", app_settings.tax.ltcg_rate)
    st.session_state.setdefault("tax_ltcg_days", app_settings.tax.ltcg_holding_days)

def go_to_step(step: int):
    st.session_state.wizard_step = step

# -------------------------------------------------------------------
# Sidebar helpers: universe presets & app settings
# -------------------------------------------------------------------

def render_universe_presets_sidebar() -> None:
    """Small CRUD UI for universe presets stored in universe_presets.yaml."""
    presets = load_universe_presets()
    if not presets:
        # This will also seed the default preset
        presets = load_universe_presets()

    preset_names = sorted(presets.keys())
    selected_name = st.selectbox(
        "Edit preset",
        options=preset_names,
        key="sidebar_universe_preset_editor",
    )

    preset = presets[selected_name]

    st.caption("Adjust the filters below and save as the same or a new name.")
    desc = st.text_input(
        "Description",
        value=preset.get("description", ""),
        key="sidebar_universe_desc",
    )
    asset_types_str = st.text_input(
        "Asset types (comma-separated)",
        value=", ".join(preset.get("asset_types", [])),
        key="sidebar_universe_asset_types",
    )
    include_cats_str = st.text_input(
        "Include categories (comma-separated; empty = all)",
        value=", ".join(preset.get("include_categories", [])),
        key="sidebar_universe_include_cats",
    )
    exclude_cats_str = st.text_input(
        "Exclude categories (comma-separated)",
        value=", ".join(preset.get("exclude_categories", [])),
        key="sidebar_universe_exclude_cats",
    )

    only_direct = st.checkbox(
        "Only direct plans",
        value=preset.get("only_direct", True),
        key="sidebar_universe_only_direct",
    )
    only_active = st.checkbox(
        "Only active (exclude index/ETF)",
        value=preset.get("only_active", True),
        key="sidebar_universe_only_active",
    )
    investible_only = st.checkbox(
        "Investible today only",
        value=preset.get("investible_only", True),
        key="sidebar_universe_investible_only",
    )
    growth_only = st.checkbox(
        "Growth option only",
        value=preset.get("growth_only", True),
        key="sidebar_universe_growth_only",
    )

    new_name = st.text_input(
        "Save as preset name",
        value=selected_name,
        key="sidebar_universe_new_name",
        help="Change this to create a new preset based on the current one.",
    )

    col_save, col_delete = st.columns(2)
    with col_save:
        if st.button("💾 Save preset", key="sidebar_universe_save"):
            name = new_name.strip()
            if not name:
                st.error("Preset name cannot be empty.")
            else:
                presets[name] = {
                    "description": desc.strip(),
                    "asset_types": [s.strip() for s in asset_types_str.split(",") if s.strip()],
                    "include_categories": [s.strip() for s in include_cats_str.split(",") if s.strip()],
                    "exclude_categories": [s.strip() for s in exclude_cats_str.split(",") if s.strip()],
                    "only_direct": only_direct,
                    "only_active": only_active,
                    "investible_only": investible_only,
                    "growth_only": growth_only,
                }
                save_universe_presets(presets)
                st.session_state.universe_preset = name
                st.success(f"Preset '{name}' saved.")

    with col_delete:
        if (
            selected_name != "equity_active_direct"
            and st.button("🗑 Delete", key="sidebar_universe_delete")
        ):
            presets.pop(selected_name, None)
            save_universe_presets(presets)
            st.success(f"Preset '{selected_name}' deleted.")


def render_settings_sidebar() -> None:
    """Edit app-level fee & tax defaults and persist them."""
    app_settings = load_app_settings()

    st.caption("Defaults used to seed the wizard's Step 5 fields.")

    fees_apply = st.checkbox(
        "Apply fees by default",
        value=app_settings.fees.apply,
        key="settings_fees_apply",
    )
    fees_bps = st.number_input(
        "Annual fee (bps)",
        min_value=0.0,
        max_value=1000.0,
        step=5.0,
        value=float(app_settings.fees.annual_bps),
        key="settings_fees_annual_bps",
    )

    tax_apply = st.checkbox(
        "Apply tax by default",
        value=app_settings.tax.apply,
        key="settings_tax_apply",
    )

    col1, col2 = st.columns(2)
    with col1:
        stcg_rate = st.number_input(
            "STCG rate (%)",
            min_value=0.0,
            max_value=50.0,
            step=0.5,
            value=float(app_settings.tax.stcg_rate),
            key="settings_tax_stcg_rate",
        )
    with col2:
        ltcg_rate = st.number_input(
            "LTCG rate (%)",
            min_value=0.0,
            max_value=50.0,
            step=0.5,
            value=float(app_settings.tax.ltcg_rate),
            key="settings_tax_ltcg_rate",
        )

    ltcg_days = st.number_input(
        "LTCG minimum holding period (days)",
        min_value=0,
        max_value=3650,
        step=1,
        value=int(app_settings.tax.ltcg_holding_days),
        key="settings_tax_ltcg_days",
    )

    if st.button("💾 Save defaults", key="settings_save_button"):
        new_settings = AppSettings(
            fees=FeeSettings(apply=fees_apply, annual_bps=fees_bps),
            tax=TaxSettings(
                apply=tax_apply,
                stcg_rate=stcg_rate,
                ltcg_rate=ltcg_rate,
                ltcg_holding_days=ltcg_days,
            ),
        )
        save_app_settings(new_settings)

        # Also push into the current wizard session so it's consistent
        st.session_state.fees_apply = fees_apply
        st.session_state.fees_annual_bps = fees_bps
        st.session_state.tax_apply = tax_apply
        st.session_state.tax_stcg_rate = stcg_rate
        st.session_state.tax_ltcg_rate = ltcg_rate
        st.session_state.tax_ltcg_days = ltcg_days

        st.success("Defaults saved. New backtests will use these values.")
# -------------------------------------------------------------------
# Build BacktestConfig from session_state
# -------------------------------------------------------------------

def build_config_from_state() -> BacktestConfig:
    mode = st.session_state.mode
    start_date = st.session_state.study_start
    end_date = st.session_state.study_end

    # Core configs
    study_window = StudyWindow(start=start_date, end=end_date)
    universe = UniverseConfig(preset=st.session_state.universe_preset)

    entry_signal = SignalConfig(
        name=st.session_state.entry_signal_name,
        direction=st.session_state.entry_signal_direction,
    )

    entry_selection = SelectionConfig(
        mode=st.session_state.entry_selection_mode,
        top_n=st.session_state.entry_top_n,
        min_funds=st.session_state.entry_min_funds,
        weight_scheme="equal",
    )

    rebalance = RebalanceConfig(
        frequency=st.session_state.rebalance_frequency
    )

    # Fees & tax
    fees = FeeConfig(
        apply=st.session_state.fees_apply,
        annual_bps=st.session_state.fees_annual_bps if st.session_state.fees_apply else 0.0,
    )
    tax = TaxConfig(
        apply=st.session_state.tax_apply,
        stcg_rate=st.session_state.tax_stcg_rate / 100.0,
        ltcg_rate=st.session_state.tax_ltcg_rate / 100.0,
        ltcg_holding_days=st.session_state.tax_ltcg_days,
    )

    # Optional separate rebalance criteria
    if st.session_state.use_separate_rebalance:
        reb_signal = SignalConfig(
            name=st.session_state.reb_signal_name,
            direction=st.session_state.reb_signal_direction,
        )
        reb_selection = SelectionConfig(
            mode=st.session_state.reb_selection_mode,
            top_n=st.session_state.reb_top_n,
            min_funds=st.session_state.reb_min_funds,
            weight_scheme="equal",
        )
    else:
        reb_signal = None
        reb_selection = None

    # Mode-specific: cohorts
    if mode == "rolling_cohort":
        cohorts = CohortConfig(
            start_frequency=st.session_state.cohort_start_frequency,
            horizon_years=st.session_state.cohort_horizon_years,
        )
    else:
        cohorts = None

    cfg = BacktestConfig(
        name=st.session_state.run_name,
        study_window=study_window,
        universe=universe,
        signal=entry_signal,
        selection=entry_selection,
        rebalance=rebalance,
        mode=mode,
        cohorts=cohorts,
        fees=fees,
        tax=tax,
        rebalance_signal=reb_signal,
        rebalance_selection=reb_selection,
        metadata={},  # free-form, for future Notion linking, etc.
    )

    return cfg


# -------------------------------------------------------------------
# Streamlit app
# -------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Backtest Engine Wizard",
        layout="wide",
    )

    init_session_state()

    st.title("📈 Mutual Fund Backtest Wizard")
    st.markdown(
        """
This wizard walks you step-by-step through setting up a **no-lookahead mutual fund backtest**.

At each step, you choose *what question you want the backtest to answer*.
We'll then form portfolios from your Postgres database and compare them to a **buy-and-hold Nifty 500 TRI** benchmark.
"""
    )

    step = st.session_state.wizard_step

    # Wizard progress
    st.sidebar.markdown("### Wizard steps")
    st.sidebar.write(f"Current step: **{step} / 6**")
    st.sidebar.button("⏮ Start over", on_click=lambda: go_to_step(1))

    # Extra sidebar tools
    st.sidebar.markdown("---")
    with st.sidebar.expander("📚 Universe presets", expanded=False):
        render_universe_presets_sidebar()

    with st.sidebar.expander("⚙️ Fee & tax defaults", expanded=False):
        render_settings_sidebar()

    # --------------------------------------------------------------
    # STEP 1 – Backtest type & time window
    # --------------------------------------------------------------
    if step == 1:
        st.header("Step 1 – Backtest type & time window")

        st.markdown(
            """
Here you decide **how many portfolios** we simulate and **over what calendar window**.

- **Single backtest** → one portfolio starting at the study start date.
- **Rolling cohorts** → a new portfolio starts every X months; each is held for a fixed horizon.
"""
        )

        col1, col2 = st.columns(2)
        with col1:
            mode_label = st.radio(
                "Backtest mode",
                options=["single", "rolling_cohort"],
                format_func=lambda x: "Single backtest" if x == "single" else "Rolling cohorts",
                key="mode",
            )

        with col2:
            st.session_state.study_start, st.session_state.study_end = st.date_input(
                "Study window (start and end dates)",
                value=(st.session_state.study_start, st.session_state.study_end),
                help="This is the overall calendar window the backtest is allowed to use.",
            )

        if st.session_state.mode == "rolling_cohort":
            st.markdown("#### Cohort settings")
            col3, col4 = st.columns(2)
            with col3:
                st.selectbox(
                    "How often should a new cohort start?",
                    options=["1M", "3M", "6M", "12M"],
                    format_func=lambda x: {
                        "1M": "Every month",
                        "3M": "Every 3 months",
                        "6M": "Every 6 months",
                        "12M": "Every year",
                    }[x],
                    key="cohort_start_frequency",
                )
            with col4:
                st.slider(
                    "Holding horizon for each cohort (years)",
                    min_value=1.0,
                    max_value=15.0,
                    step=0.5,
                    key="cohort_horizon_years",
                    help="Each cohort portfolio will be held for this long (or until the study window ends).",
                )

        st.markdown("---")
        col_prev, col_next = st.columns([1, 1])
        with col_prev:
            st.button("⬅ Previous", disabled=True)
        with col_next:
            st.button("Next ➡", on_click=lambda: go_to_step(2))

    # --------------------------------------------------------------
    # STEP 2 – Universe
    # --------------------------------------------------------------
    elif step == 2:
        st.header("Step 2 – Universe")

        st.markdown(
            """
The **universe** defines **which mutual funds are even eligible** for selection.

For now, we use a small set of **presets** that map to SQL filters inside the engine.  
(For example, `equity_active_direct` = active equity schemes, direct plans, investible today, excluding thematic/sector, etc.)
"""
        )

        # Pull available presets from YAML so anything you define
        # in the sidebar is immediately usable here.
        presets = load_universe_presets()
        preset_names = sorted(presets.keys()) if presets else ["equity_active_direct"]

        # Keep the selected preset valid if the list changed
        if st.session_state.universe_preset not in preset_names:
            st.session_state.universe_preset = preset_names[0]

        st.selectbox(
            "Universe preset",
            options=preset_names,
            key="universe_preset",
            help=(
                "Presets are defined in app/config/universe_presets.yaml and can "
                "be edited from the sidebar 'Universe presets' panel."
            ),
        )

        st.info(
            "Under the hood, the engine will query your Postgres DB for all schemes in this universe on each rebalance date."
        )

        st.markdown("---")
        col_prev, col_next = st.columns(2)
        with col_prev:
            st.button("⬅ Previous", on_click=lambda: go_to_step(1))
        with col_next:
            st.button("Next ➡", on_click=lambda: go_to_step(3))

    # --------------------------------------------------------------
    # STEP 3 – Entry signal & selection
    # --------------------------------------------------------------
    elif step == 3:
        st.header("Step 3 – Entry signal & selection")

        st.markdown(
            """
Here you tell the engine **how to score and pick funds at the start of the portfolio.**

Think of it as:  
> *“On each start date, which funds should I buy?”*
"""
        )

        st.subheader("Signal (scoring rule) for initial selection")

        col1, col2 = st.columns(2)
        with col1:
            st.text_input(
                "Signal name",
                key="entry_signal_name",
                help="Example: rank_12m_category, rank_3y_universe, perf_1y, etc. The engine maps this to a column in performance_ranking.",
            )
        with col2:
            st.radio(
                "How should scores be interpreted?",
                options=["asc", "desc"],
                format_func=lambda x: "Lower is better (ranks: 1 is best)" if x == "asc" else "Higher is better (returns / scores)",
                key="entry_signal_direction",
            )

        st.subheader("Selection rule at portfolio inception")

        st.radio(
            "Selection mode",
            options=["top_n", "all"],
            format_func=lambda x: "Top N funds" if x == "top_n" else "All eligible funds",
            key="entry_selection_mode",
        )

        if st.session_state.entry_selection_mode == "top_n":
            col3, col4 = st.columns(2)
            with col3:
                st.slider(
                    "Top N funds to pick",
                    min_value=5,
                    max_value=100,
                    step=1,
                    key="entry_top_n",
                )
            with col4:
                st.slider(
                    "Minimum eligible funds required",
                    min_value=1,
                    max_value=50,
                    step=1,
                    key="entry_min_funds",
                    help="If fewer than this many funds are eligible, the backtest will fail for that date.",
                )
        else:
            st.slider(
                "Minimum eligible funds required",
                min_value=1,
                max_value=50,
                step=1,
                key="entry_min_funds",
                help="If fewer than this many funds are eligible, the backtest will fail for that date.",
            )

        st.info("Weights are currently always equal-weight across the selected funds.")

        st.markdown("---")
        col_prev, col_next = st.columns(2)
        with col_prev:
            st.button("⬅ Previous", on_click=lambda: go_to_step(2))
        with col_next:
            st.button("Next ➡", on_click=lambda: go_to_step(4))

    # --------------------------------------------------------------
    # STEP 4 – Rebalancing & (optional) separate criteria
    # --------------------------------------------------------------
    elif step == 4:
        st.header("Step 4 – Rebalancing behaviour")

        st.markdown(
            """
Now decide **if and how the portfolio should be rebalanced over time.**

- **No rebalancing** → buy at inception and hold until the end (pure momentum / selection effect).
- **Time-based rebalancing** → periodically re-run your logic and adjust the portfolio.
"""
        )

        st.selectbox(
            "Rebalancing frequency",
            options=["NONE", "1M", "3M", "6M", "12M", "18M", "24M"],
            format_func=lambda x: {
                "NONE": "No rebalancing (buy & hold)",
                "1M": "Every month",
                "3M": "Every 3 months",
                "6M": "Every 6 months",
                "12M": "Every 12 months (annual)",
                "18M": "Every 18 months",
                "24M": "Every 24 months",
            }[x],
            key="rebalance_frequency",
        )

        if st.session_state.rebalance_frequency != "NONE":
            st.checkbox(
                "Use different criteria at rebalancing than at initial selection?",
                key="use_separate_rebalance",
            )

            if st.session_state.use_separate_rebalance:
                st.subheader("Rebalance signal & selection (used from 2nd period onwards)")
                col1, col2 = st.columns(2)
                with col1:
                    st.text_input(
                        "Rebalance signal name",
                        key="reb_signal_name",
                        help="Example: rank_12m_category, perf_3y, blended rank, etc.",
                    )
                with col2:
                    st.radio(
                        "How should rebalance scores be interpreted?",
                        options=["asc", "desc"],
                        format_func=lambda x: "Lower is better (ranks: 1 is best)" if x == "asc" else "Higher is better (returns / scores)",
                        key="reb_signal_direction",
                    )

                st.radio(
                    "Rebalance selection mode",
                    options=["top_n", "all"],
                    format_func=lambda x: "Top N funds" if x == "top_n" else "All eligible funds",
                    key="reb_selection_mode",
                )

                if st.session_state.reb_selection_mode == "top_n":
                    col3, col4 = st.columns(2)
                    with col3:
                        st.slider(
                            "Top N funds to pick at rebalance",
                            min_value=5,
                            max_value=100,
                            step=1,
                            key="reb_top_n",
                        )
                    with col4:
                        st.slider(
                            "Minimum eligible funds required at rebalance",
                            min_value=1,
                            max_value=50,
                            step=1,
                            key="reb_min_funds",
                        )
                else:
                    st.slider(
                        "Minimum eligible funds required at rebalance",
                        min_value=1,
                        max_value=50,
                        step=1,
                        key="reb_min_funds",
                    )
            else:
                st.info(
                    "From the second period onwards, the engine will reuse the **same signal and selection** as at inception."
                )
        else:
            st.info(
                "No rebalancing: we form the portfolio once at the start of its life and then leave it untouched."
            )

        st.markdown("---")
        col_prev, col_next = st.columns(2)
        with col_prev:
            st.button("⬅ Previous", on_click=lambda: go_to_step(3))
        with col_next:
            st.button("Next ➡", on_click=lambda: go_to_step(5))

    # --------------------------------------------------------------
    # STEP 5 – Costs & taxes
    # --------------------------------------------------------------
    elif step == 5:
        st.header("Step 5 – Costs & taxes")

        st.markdown(
            """
To get **realistic results**, we need to account for fees and (eventually) tax.

Right now the engine **applies fees** as a continuous drag on portfolio returns.  
Tax settings are captured here for future use when we switch to a full **position-based tax engine**.
"""
        )

        st.subheader("Fees")

        col1, col2 = st.columns(2)
        with col1:
            st.checkbox(
                "Apply an annual fee to the portfolio?",
                key="fees_apply",
            )
        with col2:
            st.slider(
                "Annual fee (basis points)",
                min_value=0.0,
                max_value=300.0,
                step=5.0,
                key="fees_annual_bps",
                help="100 bps = 1.00% p.a. This is applied as a smooth drag on returns.",
            )

        st.subheader("Tax (for future position-based engine)")

        col3, col4, col5 = st.columns(3)
        with col3:
            st.checkbox(
                "Capture capital gains tax settings?",
                key="tax_apply",
                help="These values are stored in the config but not yet applied to returns.",
            )
        with col4:
            st.number_input(
                "STCG rate (%)",
                min_value=0.0,
                max_value=50.0,
                step=0.5,
                key="tax_stcg_rate",
            )
        with col5:
            st.number_input(
                "LTCG rate (%)",
                min_value=0.0,
                max_value=50.0,
                step=0.5,
                key="tax_ltcg_rate",
            )

        st.number_input(
            "LTCG minimum holding period (days)",
            min_value=0,
            max_value=3650,
            step=1,
            key="tax_ltcg_days",
            help="Gains on holdings beyond this age will be treated as long-term for tax purposes.",
        )

        st.info(
            "For now, only fees are actually applied. Tax settings are recorded in the BacktestConfig so that future versions of the engine can honour them exactly per trade."
        )

        st.markdown("---")
        col_prev, col_next = st.columns(2)
        with col_prev:
            st.button("⬅ Previous", on_click=lambda: go_to_step(4))
        with col_next:
            st.button("Next ➡", on_click=lambda: go_to_step(6))

    # --------------------------------------------------------------
    # STEP 6 – Review & run
    # --------------------------------------------------------------
    elif step == 6:
        st.header("Step 6 – Review & run")

        st.markdown("First, give this backtest run a short, memorable name:")

        st.text_input(
            "Run name (used for CSV filenames and identification)",
            key="run_name",
        )

        cfg = build_config_from_state()

        st.subheader("Summary of your backtest setup")

        # Display a compact summary table of key parameters
        summary_rows = []

        entry_sel_mode = cfg.selection.mode
        entry_top_n = cfg.selection.top_n if entry_sel_mode == "top_n" else None

        reb_mode = cfg.rebalance_selection.mode if cfg.rebalance_selection else None
        reb_top_n = (
            cfg.rebalance_selection.top_n
            if cfg.rebalance_selection and cfg.rebalance_selection.mode == "top_n"
            else None
        )

        summary_rows.append(
            {
                "Parameter": "Backtest mode",
                "Value": "Single" if cfg.mode == "single" else "Rolling cohorts",
            }
        )
        summary_rows.append(
            {
                "Parameter": "Study window",
                "Value": f"{cfg.study_window.start} → {cfg.study_window.end}",
            }
        )
        if cfg.mode == "rolling_cohort" and cfg.cohorts is not None:
            summary_rows.append(
                {
                    "Parameter": "Cohort frequency",
                    "Value": cfg.cohorts.start_frequency,
                }
            )
            summary_rows.append(
                {
                    "Parameter": "Cohort horizon (years)",
                    "Value": cfg.cohorts.horizon_years,
                }
            )

        summary_rows.append(
            {"Parameter": "Universe preset", "Value": cfg.universe.preset}
        )
        summary_rows.append(
            {
                "Parameter": "Entry signal",
                "Value": f"{cfg.signal.name} (direction={cfg.signal.direction})",
            }
        )
        if entry_sel_mode == "top_n":
            summary_rows.append(
                {
                    "Parameter": "Entry selection",
                    "Value": f"Top {entry_top_n}, min {cfg.selection.min_funds}",
                }
            )
        else:
            summary_rows.append(
                {
                    "Parameter": "Entry selection",
                    "Value": f"All eligible, min {cfg.selection.min_funds}",
                }
            )

        summary_rows.append(
            {
                "Parameter": "Rebalance frequency",
                "Value": cfg.rebalance.frequency,
            }
        )

        if cfg.rebalance.frequency != "NONE":
            if cfg.rebalance_signal is None:
                summary_rows.append(
                    {
                        "Parameter": "Rebalance criteria",
                        "Value": "Same as entry selection",
                    }
                )
            else:
                summary_rows.append(
                    {
                        "Parameter": "Rebalance signal",
                        "Value": f"{cfg.rebalance_signal.name} (direction={cfg.rebalance_signal.direction})",
                    }
                )
                if reb_mode == "top_n":
                    summary_rows.append(
                        {
                            "Parameter": "Rebalance selection",
                            "Value": f"Top {reb_top_n}, min {cfg.rebalance_selection.min_funds}",
                        }
                    )
                else:
                    summary_rows.append(
                        {
                            "Parameter": "Rebalance selection",
                            "Value": f"All eligible, min {cfg.rebalance_selection.min_funds}",
                        }
                    )

        summary_rows.append(
            {
                "Parameter": "Fees applied?",
                "Value": f"{cfg.fees.apply} (annual_bps={cfg.fees.annual_bps})",
            }
        )
        summary_rows.append(
            {
                "Parameter": "Tax settings captured?",
                "Value": f"{cfg.tax.apply} (stcg={cfg.tax.stcg_rate*100:.1f}%, ltcg={cfg.tax.ltcg_rate*100:.1f}% after {cfg.tax.ltcg_holding_days} days)",
            }
        )

        st.table(pd.DataFrame(summary_rows))

        # English description of the research question
        question_text = describe_backtest(
            mode=cfg.mode,
            start_date=cfg.study_window.start,
            end_date=cfg.study_window.end,
            universe_preset=cfg.universe.preset,
            entry_signal_name=cfg.signal.name,
            entry_direction=cfg.signal.direction,
            entry_mode=cfg.selection.mode,
            entry_top_n=cfg.selection.top_n if cfg.selection.mode == "top_n" else None,
            rebalance_freq=cfg.rebalance.frequency,
            has_separate_rebalance=cfg.rebalance_signal is not None,
            rebalance_signal_name=cfg.rebalance_signal.name if cfg.rebalance_signal else None,
            rebalance_direction=cfg.rebalance_signal.direction if cfg.rebalance_signal else None,
            rebalance_mode=reb_mode,
            rebalance_top_n=reb_top_n,
            cohort_start_freq=cfg.cohorts.start_frequency if cfg.cohorts else None,
            cohort_horizon_years=cfg.cohorts.horizon_years if cfg.cohorts else None,
            fees_apply=cfg.fees.apply,
            fees_annual_bps=cfg.fees.annual_bps,
        )

        st.subheader("In plain English, this backtest is asking:")
        st.markdown(f"> {question_text}")

        st.markdown("---")

        col_prev, col_run = st.columns(2)
        with col_prev:
            st.button("⬅ Previous", on_click=lambda: go_to_step(5))

        run_clicked = col_run.button("🚀 Run backtest")

        if run_clicked:
            st.markdown("### Running backtest…")
            provider = PostgresDataProvider()
            engine = BacktestEngine(provider)

            try:
                result = engine.run(cfg)
            except RuntimeError as e:
                st.error(
                    f"Backtest failed: {e}\n\n"
                    "Common causes:\n"
                    "- SSH tunnel / Postgres connection not running\n"
                    "- PGPASSWORD / other PG* env vars not exported in the terminal\n"
                    "- Universe too restrictive (no eligible funds on some dates)\n"
                )
                return
            except Exception as e:
                st.exception(e)
                return

            st.success("Backtest completed successfully ✅")

            # Show summary & periods
            st.subheader("Backtest summary")
            st.dataframe(result.summary)

            with st.expander("Period-by-period breakdown"):
                st.dataframe(result.portfolio_periods)

            # Save CSVs
            paths = result.save(OUTPUT_DIR, level="standard")
            st.markdown("### Outputs saved")
            for label, path in paths.items():
                st.write(f"- **{label}** → `{path.relative_to(PROJECT_ROOT)}`")

            st.info(
                "You can now analyse these CSVs in notebooks, Excel, or link them back into your Notion Variant pages."
            )


if __name__ == "__main__":
    main()