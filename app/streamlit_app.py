from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal, Dict, List, Tuple

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
from backtest_engine.db import run_sql

from backtest_engine.app_settings import (
    load_universe_presets,
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
    if rebalance_freq == "NONE":
        rebalance_desc = "and then simply bought and held that portfolio without any rebalancing"
    else:
        if not has_separate_rebalance:
            rebalance_desc = (
                f"and then rebalanced the portfolio using the **same criteria** "
                f"{freq_map.get(rebalance_freq, 'on a fixed schedule')}"
            )
        else:
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
# Session state init
# -------------------------------------------------------------------
def init_session_state():
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 1

    # Navigation: default to wizard
    st.session_state.setdefault("nav_page", "Backtest wizard")

    # --- Basic backtest defaults ---------------------------------------
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
    st.session_state.setdefault("reb_signal_name", "rank_12m_category")
    st.session_state.setdefault("reb_signal_direction", "asc")
    st.session_state.setdefault("reb_selection_mode", "top_n")
    st.session_state.setdefault("reb_top_n", 15)
    st.session_state.setdefault("reb_min_funds", 10)

    # Cohorts
    st.session_state.setdefault("cohort_start_frequency", "1M")
    st.session_state.setdefault("cohort_horizon_years", 3.0)

    # --- Fees & taxes: HARD-CODED wizard defaults ----------------------
    # Your requested defaults:
    #   fee      = 100 bps (1.00% p.a.)
    #   STCG     = 20%
    #   LTCG     = 12.5%
    #   LTCG min = 365 days

    st.session_state.setdefault("fees_apply", True)
    st.session_state.setdefault("fees_annual_bps", 100.0)

    st.session_state.setdefault("tax_apply", True)
    st.session_state.setdefault("tax_stcg_rate", 20.0)
    st.session_state.setdefault("tax_ltcg_rate", 12.5)
    st.session_state.setdefault("tax_ltcg_days", 365)

def go_to_step(step: int):
    st.session_state.wizard_step = step


# -------------------------------------------------------------------
# Universe Presets Page (full-page manager)
# -------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_category_taxonomy() -> Tuple[List[str], List[str], Dict[str, List[str]], str | None]:
    """
    Pull distinct asset_type + category from sclass_mst.

    Returns:
        asset_types, categories, categories_by_asset_type, error_message
    """
    try:
        df = run_sql(
            """
            SELECT DISTINCT asset_type, category
            FROM sclass_mst
            ORDER BY asset_type, category;
            """
        )
    except Exception as e:
        return [], [], {}, f"{e}"

    if df.empty:
        return [], [], {}, "No rows returned from sclass_mst."

    df["asset_type"] = df["asset_type"].fillna("Unknown")
    df["category"] = df["category"].fillna("Unknown")

    asset_types = sorted(df["asset_type"].unique().tolist())
    categories = sorted(df["category"].unique().tolist())
    cats_by_asset: Dict[str, List[str]] = {}
    for atype in asset_types:
        cats = (
            df.loc[df["asset_type"] == atype, "category"]
            .dropna()
            .unique()
            .tolist()
        )
        cats_by_asset[atype] = sorted(cats)

    return asset_types, categories, cats_by_asset, None

def universe_presets_page() -> None:
    st.title("📚 Universe presets")

    st.markdown(
        """
Define **named universes** that control which mutual funds are even eligible for selection.

Each preset stores:

- A label + description (for humans).
- Filters like:
  - **Asset types** (e.g. Equity, Hybrid, Debt).
  - **Include / exclude categories** (backed by `sclass_mst`).
- Flags such as:
  - Only direct plans
  - Only active funds (exclude pure index/ETFs)
  - Investible today only
  - Growth option only
"""
    )

    # Load presets from YAML (and seed defaults if needed)
    presets = load_universe_presets()
    if not presets:
        presets = load_universe_presets()

    preset_names = sorted(presets.keys())

    # Load DB taxonomy for asset_type / category
    asset_types_all, categories_all, cats_by_asset, taxonomy_error = load_category_taxonomy()

    if taxonomy_error:
        st.warning(
            "Could not load `asset_type` / `category` list from DB. "
            "You can still edit presets, but category options are not validated.\n\n"
            f"Raw error: `{taxonomy_error}`"
        )

    tab_list, tab_edit = st.tabs(["📋 Preset list", "✏️ Create / edit preset"])

    # ------------------------------------------------------------------
    # TAB 1 – List all presets in a neat table
    # ------------------------------------------------------------------
    with tab_list:
        st.subheader("Existing presets")

        if not preset_names:
            st.info("No presets yet. Switch to the **Create / edit preset** tab to add one.")
        else:
            rows = []
            for name in preset_names:
                p = presets.get(name, {})
                rows.append(
                    {
                        "key": name,
                        "label": p.get("label", name.replace("_", " ").title()),
                        "asset_types": ", ".join(p.get("asset_types", [])) or "—",
                        "include_categories": ", ".join(p.get("include_categories", [])) or "—",
                        "exclude_categories": ", ".join(p.get("exclude_categories", [])) or "—",
                        "flags": ", ".join(
                            [
                                text
                                for flag, text in [
                                    (p.get("only_direct", False), "direct-only"),
                                    (p.get("only_active", False), "active-only"),
                                    (p.get("investible_only", False), "investible-only"),
                                    (p.get("growth_only", False), "growth-only"),
                                ]
                                if flag
                            ]
                        )
                        or "—",
                    }
                )

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "To modify any preset, switch to the **Create / edit preset** tab and select it there."
            )

    # ------------------------------------------------------------------
    # TAB 2 – Create / edit presets (clean form layout)
    # ------------------------------------------------------------------
    with tab_edit:
        st.subheader("Pick or create a preset")

        # If any button set a "pending" choice last run, apply it
        if "uni_choice_pending" in st.session_state:
            st.session_state["uni_choice"] = st.session_state.pop("uni_choice_pending")

        options = ["<New preset>"] + preset_names
        choice = st.selectbox(
            "Preset to edit",
            options=options,
            key="uni_choice",
            help="Choose an existing preset to edit, or '<New preset>' to start from a blank template.",
        )

        if choice == "<New preset>":
            original_name = None
            working = {
                "label": "",
                "description": "",
                "asset_types": ["Equity"] if "Equity" in asset_types_all else [],
                "include_categories": [],
                "exclude_categories": [],
                "only_direct": True,
                "only_active": True,
                "investible_only": True,
                "growth_only": True,
            }
        else:
            original_name = choice
            working = presets[choice].copy()

        st.markdown("### Basic info")

        col_basic_1, col_basic_2 = st.columns(2)
        with col_basic_1:
            internal_name = st.text_input(
                "Preset key (internal)",
                value=original_name or "",
                key="uni_internal_name",
                help="Short identifier like 'equity_active_direct'. Used in configs & backtests.",
            )
        with col_basic_2:
            label = st.text_input(
                "Label (for UI display)",
                value=working.get(
                    "label",
                    internal_name.replace("_", " ").title() if internal_name else "",
                ),
                key="uni_label",
            )

        desc = st.text_area(
            "Description",
            value=working.get("description", ""),
            key="uni_desc",
        )

        st.markdown("### Filters")

        # --- Asset types & categories side by side ----------------------
        col_filters_left, col_filters_right = st.columns(2)

        with col_filters_left:
            # Asset types
            existing_asset_types = working.get("asset_types", []) or []
            asset_options = sorted(set(asset_types_all) | set(existing_asset_types))

            if not asset_options:
                asset_options = existing_asset_types  # fallback

            default_assets = existing_asset_types
            if not default_assets and "Equity" in asset_options:
                default_assets = ["Equity"]
            elif not default_assets:
                default_assets = asset_options

            selected_asset_types = st.multiselect(
                "Asset types",
                options=asset_options,
                default=default_assets,
                help="Leave empty to cover all asset types in the DB.",
                key="uni_asset_types",
            )

            # Include categories
            existing_inc = working.get("include_categories", []) or []
            cat_options = sorted(
                set(categories_all) | set(existing_inc) | set(working.get("exclude_categories", []))
            )

            include_categories = st.multiselect(
                "Include categories (optional)",
                options=cat_options,
                default=existing_inc,
                help="If left empty, all categories under the chosen asset types are allowed.",
                key="uni_include_categories",
            )

        with col_filters_right:
            # Exclude categories
            existing_exc = working.get("exclude_categories", []) or []
            cat_options = sorted(
                set(categories_all) | set(existing_exc) | set(working.get("include_categories", []))
            )

            exclude_categories = st.multiselect(
                "Exclude categories",
                options=cat_options,
                default=existing_exc,
                help="Categories to *drop* from the universe (e.g. Sectoral / Thematic, ELSS, Solution Oriented, Liquid, Overnight).",
                key="uni_exclude_categories",
            )

            st.markdown("#### Flags")
            only_direct = st.checkbox(
                "Only direct plans",
                value=working.get("only_direct", True),
                key="uni_only_direct",
            )
            only_active = st.checkbox(
                "Only active funds (exclude pure index/ETF)",
                value=working.get("only_active", True),
                key="uni_only_active",
            )
            investible_only = st.checkbox(
                "Investible today only",
                value=working.get("investible_only", True),
                key="uni_investible_only",
            )
            growth_only = st.checkbox(
                "Growth option only (dividendoptionflag = 'Z')",
                value=working.get("growth_only", True),
                key="uni_growth_only",
            )

        # Optional DB taxonomy preview
        if asset_types_all and cats_by_asset:
            with st.expander("🔎 DB taxonomy: asset types & categories", expanded=False):
                rows = []
                for atype, cats in cats_by_asset.items():
                    for cat in cats:
                        rows.append({"asset_type": atype, "category": cat})
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("### Actions")

        col_save, col_delete = st.columns([2, 1])

        with col_save:
            if st.button("💾 Save preset", key="uni_save_button"):
                name = internal_name.strip()
                if not name:
                    st.error("Preset key cannot be empty.")
                else:
                    if original_name and name != original_name and name in presets:
                        st.error(f"Preset key '{name}' already exists.")
                    else:
                        payload = {
                            "label": label.strip() or name.replace("_", " ").title(),
                            "description": desc.strip(),
                            "asset_types": selected_asset_types,
                            "include_categories": include_categories,
                            "exclude_categories": exclude_categories,
                            "only_direct": bool(only_direct),
                            "only_active": bool(only_active),
                            "investible_only": bool(investible_only),
                            "growth_only": bool(growth_only),
                        }

                        # Handle rename
                        if original_name and name != original_name:
                            presets.pop(original_name, None)

                        presets[name] = payload
                        save_universe_presets(presets)

                        st.session_state.universe_preset = name
                        st.session_state["uni_choice_pending"] = name
                        st.success(f"Preset '{name}' saved.")

        with col_delete:
            can_delete = original_name is not None
            if st.button(
                "🗑 Delete preset",
                key="uni_delete_button",
                disabled=not can_delete,
                help="You cannot delete while creating a brand new preset.",
            ):
                if original_name is not None:
                    presets.pop(original_name, None)
                    save_universe_presets(presets)
                    st.session_state["uni_choice_pending"] = "<New preset>"
                    st.success(f"Preset '{original_name}' deleted.")

        st.markdown("---")
        st.markdown("### Effective filter summary")

        summary_lines = [
            f"- **Key**: `{internal_name or '—'}`",
            f"- **Label**: {label or '—'}",
            f"- **Asset types**: {', '.join(selected_asset_types) if selected_asset_types else 'All asset types'}",
        ]
        if include_categories:
            summary_lines.append(
                f"- **Include categories**: {', '.join(include_categories)}"
            )
        else:
            summary_lines.append(
                "- **Include categories**: All categories under the asset types above"
            )

        if exclude_categories:
            summary_lines.append(
                f"- **Exclude categories**: {', '.join(exclude_categories)}"
            )

        flags_text = []
        if only_direct:
            flags_text.append("Direct plans only")
        if only_active:
            flags_text.append("Active funds only")
        if investible_only:
            flags_text.append("Investible today only")
        if growth_only:
            flags_text.append("Growth option only")

        summary_lines.append(
            "- **Flags**: " + (", ".join(flags_text) if flags_text else "None")
        )

        st.markdown("\n".join(summary_lines))
# -------------------------------------------------------------------
# App Settings Page (fees & tax defaults)
# -------------------------------------------------------------------

def app_settings_page() -> None:
    st.title("⚙️ App settings – fee & tax defaults")

    st.markdown(
        """
These settings control the **default values** that the wizard uses on
**Step 5 – Costs & taxes**.

You can override them per-run inside the wizard; this page just sets
the *starting defaults* for new sessions.
"""
    )

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

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        options=["Backtest wizard", "Universe presets"],
        key="nav_page",
    )

    # Route to dedicated pages first
    if page == "Universe presets":
        universe_presets_page()
        return

    # From here on, we're in the Backtest wizard
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
            st.radio(
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

We use **presets** that map to SQL filters inside the engine.  
(For example, `equity_active_direct` = active equity schemes, direct plans, investible today, excluding thematic/sector, etc.)

You can define and edit presets on the **Universe presets** page in the sidebar.
"""
        )

        # Pull available presets from YAML so anything you define
        # on the Universe presets page is immediately usable here.
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
                "be edited from the 'Universe presets' page."
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