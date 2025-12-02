# Directory Tree (Python & Notebook Files Only)

```
./
    all_code.py
    app/
        streamlit_app.py
        config/
        backtests/
            mf-10y-annual-top15-net1%/
    backtests/
    notebooks/
        02_frontend.ipynb
        01_smoke_test.ipynb
        batch_outputs/
            mf-10y-annual-top30/
            mf-10y-annual-top15/
            frontend_examples/
                mf-10y-annual-top30-net1%/
                mf-10y-annual-top15-net1%/
                mf-10y-semiannual-top15-net1%/
        outputs/
            single_runs/
    src/
        backtest_engine.egg-info/
        backtest_engine/
            db.py
            app_settings.py
            config.py
            formula.py
            presets.py
            postgres_provider.py
            batch.py
            __init__.py
            engine.py
            data_provider.py
            selection.py
            utils/
                __init__.py
                dates.py
                __pycache__/
            __pycache__/
            data/
                market/
```

## all_code.py

```python
import os
import json

# Function to extract Python code from .py or .ipynb
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


# Function to generate a directory tree (ONLY .py and .ipynb, ignoring hidden folders)
def generate_tree(root_dir):
    tree_lines = []

    for root, dirs, files in os.walk(root_dir):

        # Remove hidden folders (anything starting with .)
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        # Folder entry
        level = root.replace(root_dir, "").count(os.sep)
        indent = "    " * level
        folder_name = os.path.basename(root) if root != root_dir else root_dir
        tree_lines.append(f"{indent}{folder_name}/")

        # Only include .py and .ipynb files (and skip hidden files)
        sub_indent = "    " * (level + 1)
        for f in files:
            if not f.startswith(".") and f.endswith((".py", ".ipynb")):
                tree_lines.append(f"{sub_indent}{f}")

    return "\n".join(tree_lines)


# Create Markdown file containing tree + extracted code
def create_markdown_from_code(root_dir, output_md):
    with open(output_md, 'w', encoding='utf-8') as md_file:

        # Write directory tree at the top
        tree_output = generate_tree(root_dir)
        md_file.write("# Directory Tree (Python & Notebook Files Only)\n\n")
        md_file.write("```\n")
        md_file.write(tree_output)
        md_file.write("\n```\n\n")

        # Extraction phase
        for root, dirs, files in os.walk(root_dir):

            # Remove hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for file in files:
                if not file.startswith(".") and file.endswith((".py", ".ipynb")):
                    file_path = os.path.join(root, file)
                    code = extract_code_from_file(file_path)

                    if not code.strip():
                        continue

                    md_file.write(f"## {file}\n\n")
                    md_file.write("```python\n")
                    md_file.write(code)
                    md_file.write("\n```\n\n")


# Run
root_directory = "."
output_markdown = "all_code.md"
create_markdown_from_code(root_directory, output_markdown)

print(f"Markdown file '{output_markdown}' created.")
```

## streamlit_app.py

```python
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

from backtest_engine.db import run_sql, get_engine
from backtest_engine.formula import load_selection_field_registry

from backtest_engine.app_settings import (
    load_universe_presets,
    save_universe_presets,
)

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
    # Signal definition mode + formula bits
    st.session_state.setdefault("entry_signal_mode", "simple")  # "simple" or "formula"
    st.session_state.setdefault("entry_signal_expression", "")
    st.session_state.setdefault("entry_signal_filter_expression", "")
    st.session_state.setdefault("entry_signal_field_search", "")

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
def load_signal_field_registry():
    """
    Cached wrapper over load_selection_field_registry for the Step 3 UI.

    Returns (registry, error_msg). 'registry' maps column_name -> FieldInfo,
    and FieldInfo.table tells you which table it came from.
    """
    try:
        engine = get_engine()
        registry = load_selection_field_registry(engine)
    except Exception as e:
        return {}, str(e)

    return registry, None

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

@st.cache_data(show_spinner=False)
def load_signal_fields() -> tuple[list[str], str | None]:
    """
    Load the list of allowed fields from performance_ranking
    that can be used in Step 3 formulas.
    """
    try:
        engine = get_engine()
        registry = load_selection_field_registry(engine)
    except Exception as e:
        return [], str(e)

    return sorted(registry.keys()), None

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

    # Entry signal: simple vs formula mode
    entry_signal_mode = st.session_state.get("entry_signal_mode", "simple")
    expr = st.session_state.get("entry_signal_expression", "").strip()
    filt_expr = st.session_state.get("entry_signal_filter_expression", "").strip()

    # Only keep expressions if we're in formula mode and something is actually entered
    if entry_signal_mode != "formula":
        expr = None
        filt_expr = None
    else:
        if not expr:
            expr = None
        if not filt_expr:
            filt_expr = None

    entry_signal = SignalConfig(
        name=st.session_state.entry_signal_name,
        direction=st.session_state.entry_signal_direction,
        expression=expr,
        filter_expression=filt_expr,
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
> *“On each start date, which funds should I buy, and based on what score?”*
"""
        )

        # --- handle any "insert column into formula" clicks -------------
        # (buttons set a pending value, we apply it before rendering widgets)
        if "entry_signal_expr_append" in st.session_state:
            to_add = st.session_state.pop("entry_signal_expr_append")
            cur = st.session_state.get("entry_signal_expression", "")
            # Add a space if needed (avoid squashing tokens)
            sep = "" if not cur or cur.endswith((" ", "(", "+", "-", "*", "/", ",")) else " "
            st.session_state["entry_signal_expression"] = cur + sep + to_add

        if "entry_signal_filter_append" in st.session_state:
            to_add = st.session_state.pop("entry_signal_filter_append")
            cur = st.session_state.get("entry_signal_filter_expression", "")
            sep = "" if not cur or cur.endswith((" ", "(", "+", "-", "*", "/", ",")) else " "
            st.session_state["entry_signal_filter_expression"] = cur + sep + to_add

        # ----------------- SIGNAL DEFINITION -------------------------
        st.subheader("Signal (scoring rule) for initial selection")

        signal_mode = st.radio(
            "How do you want to define the signal?",
            options=["simple", "formula"],
            format_func=lambda x: "Simple named signal" if x == "simple" else "Custom formula over DB fields",
            key="entry_signal_mode",
        )

        if signal_mode == "simple":
            # Existing behaviour: just a name that maps to a single column
            col1, col2 = st.columns(2)
            with col1:
                st.text_input(
                    "Signal name",
                    key="entry_signal_name",
                    help=(
                        "Example: rank_12m_category, rank_3y_universe, perf_1y, etc. "
                        "The engine maps this to a single column in performance_ranking."
                    ),
                )
            with col2:
                st.radio(
                    "How should scores be interpreted?",
                    options=["asc", "desc"],
                    format_func=lambda x: "Lower is better (ranks: 1 is best)" if x == "asc"
                    else "Higher is better (returns / scores)",
                    key="entry_signal_direction",
                )

            st.caption(
                "In this mode the signal name is mapped to one column, e.g. "
                "`rank_12m_category` → `rank_1y_category` in performance_ranking."
            )

        else:
            # Formula mode: score = expression(performance_ranking fields)
            col1, col2 = st.columns(2)
            with col1:
                st.text_input(
                    "Logical name for this signal",
                    key="entry_signal_name",
                    help=(
                        "A short label like 'mom_blend_1_3_5y'. "
                        "This is stored in the config and used in filenames."
                    ),
                )
            with col2:
                st.radio(
                    "How should scores be interpreted?",
                    options=["asc", "desc"],
                    format_func=lambda x: "Lower is better (ranks: 1 is best)" if x == "asc"
                    else "Higher is better (returns / scores)",
                    key="entry_signal_direction",
                )

            st.markdown("### Score formula")

            st.text_area(
                "Score formula (score = ...)",
                key="entry_signal_expression",
                height=100,
                help=(
                    "Use columns from performance_ranking like perf_1m, perf_3m, perf_6m, perf_1y, "
                    "rank_1y_category, aum_cr, age_years, etc.\n\n"
                    "Examples:\n"
                    "  0.4 * perf_1y + 0.4 * perf_3y + 0.2 * perf_5y\n"
                    "  zscore(perf_1y) + zscore(perf_3y)\n\n"
                    "Allowed functions: abs, log, exp, sqrt, zscore, min, max, pow.\n"
                    "You can use ^ as power: perf_6m ^ 2 == perf_6m ** 2."
                ),
            )

            st.markdown("### Optional filter (pre-selection)")

            st.text_area(
                "Filter expression (boolean)",
                key="entry_signal_filter_expression",
                height=80,
                help=(
                    "Filter out funds before scoring. Example:\n"
                    "  aum_cr >= 300 and age_years >= 3\n"
                    "Only rows where this is True remain eligible."
                ),
            )

            # --- Redash-style column search + insert ------------------
                        # --- Table + column search & insert ----------------------
            registry, reg_err = load_signal_field_registry()
            with st.expander("🔎 Available tables & columns to use in formulas", expanded=False):
                if reg_err:
                    st.warning(
                        "Could not introspect selection fields from DB.\n\n"
                        f"Raw error: `{reg_err}`"
                    )
                elif not registry:
                    st.info("No fields loaded (is the DB reachable?).")
                else:
                    # Available tables from the registry (e.g. performance_ranking, scheme_details)
                    all_tables = sorted({fi.table for fi in registry.values()})

                    tables_key = "entry_signal_tables"

                    # On first render (or if somehow empty), default to "all tables"
                    if tables_key not in st.session_state or not st.session_state[tables_key]:
                        st.session_state[tables_key] = all_tables

                    selected_tables = st.multiselect(
                        "Tables to show columns from",
                        options=all_tables,
                        key=tables_key,
                        help="Pick one or more logical tables. The columns list below is restricted to these.",
                    )
                    
                    search = st.text_input(
                        "Search columns (type to filter, then click to insert)",
                        key="entry_signal_field_search",
                        placeholder="e.g. perf_1y, rank_1y_category, aum_cr ...",
                    )

                    # Filter registry by selected tables + search
                    filtered_items = [
                        (name, fi)
                        for name, fi in registry.items()
                        if (not selected_tables or fi.table in selected_tables)
                    ]
                    if search:
                        s_lower = search.lower()
                        filtered_items = [
                            (name, fi)
                            for name, fi in filtered_items
                            if s_lower in name.lower()
                        ]

                    if not filtered_items:
                        st.write("No columns match this selection.")
                    else:
                        st.caption(
                            "Click a button to insert the column name into the score or filter formula."
                        )
                        # Group by table for display
                        by_table: dict[str, list[str]] = {}
                        for name, fi in filtered_items:
                            by_table.setdefault(fi.table, []).append(name)

                        for tbl in sorted(by_table.keys()):
                            st.markdown(f"**{tbl}**")
                            for field in sorted(by_table[tbl]):
                                c1, c2, c3 = st.columns([3, 1, 1])
                                with c1:
                                    st.code(field, language="text")
                                with c2:
                                    if st.button("↳ score", key=f"expr_{tbl}_{field}"):
                                        st.session_state["entry_signal_expr_append"] = field
                                with c3:
                                    if st.button("↳ filter", key=f"filt_{tbl}_{field}"):
                                        st.session_state["entry_signal_filter_append"] = field
                                        
        # ----------------- SELECTION RULE ----------------------------
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
# src/backtest_engine/db.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------
# Local dev defaults
# ---------------------------------------------------------------------

# These are safe *for your machine* but obviously shouldn't be kept
# if this repo becomes public. They make life easy for you + your team.
DEFAULT_DB_ENV = {
    "PGHOST": "localhost",
    "PGPORT": "5433",
    "PGDATABASE": "kairo_production",
    "PGUSER": "sanjay_readonly",
    "PGPASSWORD": "Piper112358!",
}

for key, value in DEFAULT_DB_ENV.items():
    if not os.getenv(key):
        os.environ[key] = value


# ---------------------------------------------------------------------
# Engine + helpers
# ---------------------------------------------------------------------

_engine: Optional[Engine] = None


def get_engine() -> Engine:
    """Return a singleton SQLAlchemy engine configured from env vars."""
    global _engine
    if _engine is not None:
        return _engine

    PGHOST = os.getenv("PGHOST", "localhost")
    PGPORT = os.getenv("PGPORT", "5433")
    PGDATABASE = os.getenv("PGDATABASE", "kairo_production")
    PGUSER = os.getenv("PGUSER", "sanjay_readonly")
    PGPASSWORD = os.getenv("PGPASSWORD")

    if not PGPASSWORD:
        # This should rarely happen now because of DEFAULT_DB_ENV,
        # but keep the guardrail for safety.
        raise RuntimeError(
            "PGPASSWORD not found. Export it in your terminal or set DEFAULT_DB_ENV appropriately."
        )

    uri = (
        f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}"
        f"@{PGHOST}:{PGPORT}/{PGDATABASE}?sslmode=require"
    )

    _engine = create_engine(uri)
    return _engine


def run_sql(query: str, params: dict | None = None) -> pd.DataFrame:
    """Run a SQL query and return a DataFrame."""
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


def check_connection() -> bool:
    """Quick sanity check that DB is reachable."""
    try:
        df = run_sql("SELECT now() AT TIME ZONE 'Asia/Kolkata' AS now_ist;")
    except Exception:
        return False
    return not df.empty
```

## app_settings.py

```python
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict

import yaml

# ---------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------

# src/backtest_engine/app_settings.py -> project root is 2 levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# We keep config files under app/config so they live with the app
SETTINGS_PATH = PROJECT_ROOT / "app" / "config" / "app_settings.yaml"
UNIVERSES_PATH = PROJECT_ROOT / "app" / "config" / "universe_presets.yaml"


# ---------------------------------------------------------------------
# Fee & tax settings (app-level defaults)
# ---------------------------------------------------------------------

@dataclass
class FeeSettings:
    apply: bool = True
    annual_bps: float = 100.0  # 1% p.a. drag


@dataclass
class TaxSettings:
    apply: bool = False
    # Stored as percentages to match the Streamlit wizard UI
    stcg_rate: float = 15.0     # 15%
    ltcg_rate: float = 10.0     # 10%
    ltcg_holding_days: int = 365


@dataclass
class AppSettings:
    """Top-level app config.

    Right now this only carries fee & tax defaults, but you can add more
    later (e.g. default study window, default universe, etc.).
    """
    fees: FeeSettings = field(default_factory=FeeSettings)
    tax: TaxSettings = field(default_factory=TaxSettings)


# ---------------------------------------------------------------------
# Load/save app_settings.yaml
# ---------------------------------------------------------------------

def load_app_settings() -> AppSettings:
    """Load app-level settings from YAML, seeding defaults if missing."""
    if not SETTINGS_PATH.exists():
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        settings = AppSettings()
        save_app_settings(settings)
        return settings

    with SETTINGS_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    fees_data = data.get("fees", {}) or {}
    tax_data = data.get("tax", {}) or {}

    fees = FeeSettings(
        apply=bool(fees_data.get("apply", True)),
        annual_bps=float(fees_data.get("annual_bps", 100.0)),
    )
    tax = TaxSettings(
        apply=bool(tax_data.get("apply", False)),
        stcg_rate=float(tax_data.get("stcg_rate", 15.0)),
        ltcg_rate=float(tax_data.get("ltcg_rate", 10.0)),
        ltcg_holding_days=int(tax_data.get("ltcg_holding_days", 365)),
    )

    return AppSettings(fees=fees, tax=tax)


def save_app_settings(settings: AppSettings) -> None:
    """Persist app-level settings to YAML."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SETTINGS_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "fees": asdict(settings.fees),
                "tax": asdict(settings.tax),
            },
            f,
            sort_keys=False,
        )


# ---------------------------------------------------------------------
# Universe presets (universe_presets.yaml)
# ---------------------------------------------------------------------

def load_universe_presets() -> Dict[str, dict]:
    """Return mapping: preset_name -> preset_dict.

    Each preset dict is free-form but we currently use:
      - description: human readable
      - asset_types: list[str]
      - include_categories: list[str]
      - exclude_categories: list[str]
      - only_direct: bool
      - only_active: bool
      - investible_only: bool
      - growth_only: bool

    The engine today only uses `universe_config.preset`, but these
    fields are here so we can later hook them into SQL filters in
    PostgresDataProvider.get_universe().
    """
    if not UNIVERSES_PATH.exists():
        UNIVERSES_PATH.parent.mkdir(parents=True, exist_ok=True)
        presets = {
            "equity_active_direct": {
                "description": (
                    "Equity, Active, Direct, investible, growth – your current "
                    "default equity universe (ex-index/ETF, ex-sector/thematic/etc.)."
                ),
                "asset_types": ["Equity"],
                "include_categories": [],
                "exclude_categories": [],
                "only_direct": True,
                "only_active": True,
                "investible_only": True,
                "growth_only": True,
            }
        }
        save_universe_presets(presets)
        return presets

    with UNIVERSES_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Stored as: {"universes": {name: {...}, ...}}
    return data.get("universes", {})


def save_universe_presets(presets: Dict[str, dict]) -> None:
    """Persist all universe presets to YAML."""
    UNIVERSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with UNIVERSES_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"universes": presets}, f, sort_keys=False)
```

## config.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Literal, Optional

# ---------------------------------------------------------------------
# Core enums / type aliases
# ---------------------------------------------------------------------

RebalanceFreq = Literal["NONE", "1M", "3M", "6M", "12M", "18M", "24M"]
BacktestMode = Literal["single", "rolling_cohort"]


# ---------------------------------------------------------------------
# Study window / universe
# ---------------------------------------------------------------------

@dataclass
class StudyWindow:
    start: Optional[date] = None   # None = auto_min (not used yet)
    end: Optional[date] = None     # None = auto_max (not used yet)


@dataclass
class UniverseConfig:
    # Either use a preset name OR raw filters (we'll implement filters later)
    preset: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Signal / selection / rebalancing
# ---------------------------------------------------------------------

@dataclass
class SignalConfig:
    """Configuration for how to compute the ranking score.

    Two modes:

    1) Simple column mode (current behaviour):
       - Use `name`, `source`, `lookback_months`, `rank_scope`, etc.
       - Leave `expression` and `filter_expression` as None.
       - DataProvider.get_signal_scores() will resolve a single column
         via _resolve_signal_column() and use that as `score`.

    2) Expression mode:
       - Set `expression` to a formula using columns in performance_ranking,
         e.g. "0.5 * perf_1y + 0.5 * perf_3y".
       - Optionally set `filter_expression` to a boolean formula to filter funds,
         e.g. "aum_cr > 300 and age_years > 3".
       - `name` becomes just a logical label for this signal.

    direction:
       - "asc": lower raw value is better (e.g. rank_1y_category).
       - "desc": higher raw value is better (e.g. perf_1y).
       Internally we normalise so that *higher score is always better*.
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
    """Configuration for WHEN rebalancing happens.

    frequency:
      - "NONE" => buy-and-hold (single portfolio from start to end).
      - "1M", "3M", "6M", ... => rebalance on a fixed calendar schedule.

    anchor:
      - currently informational; we always anchor to the provided start date.
    """
    frequency: RebalanceFreq = "12M"
    anchor: Literal["month_end"] = "month_end"


# ---------------------------------------------------------------------
# Fee & tax config (tax still plumbing-only in engine)
# ---------------------------------------------------------------------

@dataclass
class FeeConfig:
    apply: bool = False
    annual_bps: float = 0.0   # 100 = 1.00% p.a.
    apply_frequency: Literal["daily", "monthly", "annual"] = "daily"


@dataclass
class TaxConfig:
    apply: bool = False
    stcg_rate: float = 0.15
    ltcg_rate: float = 0.10
    ltcg_holding_days: int = 365


# ---------------------------------------------------------------------
# Cohort (rolling) config
# ---------------------------------------------------------------------

@dataclass
class CohortConfig:
    """Controls rolling-cohort experiments.

    Example: monthly cohorts with 3-year holding horizon.

    start_frequency:
      - How often we start a new cohort (usually "1M" or "3M").

    horizon_years:
      - Target holding horizon for each cohort; internally converted to months.

    Notes:
      - Engine will truncate the last cohort if it would run past study_window.end.
      - For now we always reuse the same rebalance rules *inside* each cohort.
    """
    start_frequency: Literal["1M", "3M", "6M", "12M"] = "1M"
    horizon_years: float = 3.0


# ---------------------------------------------------------------------
# Top-level backtest config
# ---------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Top-level config object for a single logical backtest.

    This is what a Notion Variant row should map to.

    Modes:
      - mode="single":  traditional backtest over study_window, with optional
                        rebalancing according to `rebalance`.
      - mode="rolling_cohort": rolling cohorts inside study_window; each cohort
                               uses the same universe/signal/selection rules,
                               but with its own cohort [start, start+horizon).

    For rolling_cohort mode you must provide `cohorts`.
    """

    # --- required core pieces (no defaults) ---
    name: str
    study_window: StudyWindow
    universe: UniverseConfig
    signal: SignalConfig
    selection: SelectionConfig
    rebalance: RebalanceConfig

    # --- mode / cohorts ---
    mode: BacktestMode = "single"
    cohorts: Optional[CohortConfig] = None

    # --- fees & tax ---
    fees: FeeConfig = field(default_factory=FeeConfig)
    tax: TaxConfig = field(default_factory=TaxConfig)

    # --- optional separate rebalance criteria ---
    # If provided, these are used from the second period onwards.
    # The *initial* portfolio always uses `signal` + `selection`.
    rebalance_signal: Optional[SignalConfig] = None
    rebalance_selection: Optional[SelectionConfig] = None

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

# Which tables feed the SELECTION "mart"
SELECTION_MART_TABLES = [
    "performance_ranking",  # time-series ranking/perf snapshot
    "scheme_details",       # static scheme metadata
]


def load_selection_field_registry(
    engine: Engine,
    schema: str = "public",
) -> Dict[str, FieldInfo]:
    """
    Load the registry of *allowed* fields for SELECTION formulas.

    We build a logical "selection mart" by unioning columns from a small,
    curated set of tables (SELECTION_MART_TABLES). For now:
        - performance_ranking
        - scheme_details

    Later you can add more tables here once get_signal_scores joins them.
    """
    fields: Dict[str, FieldInfo] = {}

    for table in SELECTION_MART_TABLES:
        try:
            table_fields = load_table_columns(engine, table_name=table, schema=schema)
        except Exception:
            # If a table is missing or fails introspection, just skip it.
            continue

        for col_name, info in table_fields.items():
            # If multiple tables share a column name, keep the first one we saw.
            # In practice, perf_ranking will be first so its columns win.
            if col_name in fields:
                continue
            fields[col_name] = info

    if not fields:
        raise RuntimeError(
            f"No selection fields found in tables: {', '.join(SELECTION_MART_TABLES)}"
        )

    return fields

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

## presets.py

```python
# src/backtest_engine/presets.py

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Type, TypeVar

import json

from .config import (
    UniverseConfig,
    SignalConfig,
    SelectionConfig,
    RebalanceConfig,
    RebalanceFreq,
)

# ----------------------------------------------------------------------
# Universe presets
# ----------------------------------------------------------------------


@dataclass
class UniversePreset:
    """Named universe definition.

    This is a *UI-level* object that standardises how we describe a universe.
    It maps cleanly onto UniverseConfig (which is what the engine uses).

    Typical filters you might put here (inside `filters`):
      - asset_classes: ["Equity"]
      - include_categories: ["Flexi Cap", "Large Cap", ...]
      - exclude_categories: ["Sectoral / Thematic", "ELSS", ...]
      - direct_only: True
      - active_only: True
      - min_aum_cr: 300.0
    """

    name: str                      # internal key, e.g. "equity_active_direct"
    label: str                     # human label, e.g. "Equity – Active – Direct"
    description: str = ""          # longer explanation (for UI hover/help)
    filters: Dict[str, Any] = field(default_factory=dict)

    def to_universe_config(self) -> UniverseConfig:
        """Convert this preset into a UniverseConfig used by the engine."""
        return UniverseConfig(
            preset=self.name,
            filters=self.filters.copy(),
        )


# ----------------------------------------------------------------------
# Criteria / scoring presets
# ----------------------------------------------------------------------


@dataclass
class CriteriaPreset:
    """Named scoring / filter logic over performance_ranking.

    This is essentially a higher-level wrapper around SignalConfig:
      - `expression` and `filter_expression` are formulas interpreted by
        the formula engine against `performance_ranking` columns.
      - `direction` decides whether higher or lower scores are better.

    Example:
      name="mom_blend_1_3_5y"
      expression="0.4 * perf_1y + 0.4 * perf_3y + 0.2 * perf_5y"
      filter_expression="aum_cr > 300 and age_years >= 3"
      direction="desc"
    """

    name: str
    label: str
    description: str = ""

    # These mirror SignalConfig
    source: str = "performance_ranking"
    lookback_months: int = 12
    direction: Literal["asc", "desc"] = "desc"
    rank_scope: Literal["category", "asset_class", "universe"] = "category"
    tie_breaker: Optional[str] = None

    # Formula parts
    expression: Optional[str] = None
    filter_expression: Optional[str] = None

    def to_signal_config(self) -> SignalConfig:
        """Convert this preset into a SignalConfig for the engine."""
        return SignalConfig(
            name=self.name,
            source=self.source,
            lookback_months=self.lookback_months,
            direction=self.direction,
            rank_scope=self.rank_scope,
            tie_breaker=self.tie_breaker,
            expression=self.expression,
            filter_expression=self.filter_expression,
        )


# ----------------------------------------------------------------------
# Selection presets (top-N / all)
# ----------------------------------------------------------------------


@dataclass
class SelectionPreset:
    """Named selection rule: how many funds to pick, and with what weighting."""

    name: str
    label: str
    description: str = ""

    # These mirror SelectionConfig
    mode: Literal["top_n", "all"] = "top_n"
    top_n: int = 15
    min_funds: int = 10
    weight_scheme: Literal["equal"] = "equal"

    def to_selection_config(self) -> SelectionConfig:
        return SelectionConfig(
            mode=self.mode,
            top_n=self.top_n,
            min_funds=self.min_funds,
            weight_scheme=self.weight_scheme,
        )


# ----------------------------------------------------------------------
# Rebalance presets (time-based for now)
# ----------------------------------------------------------------------


@dataclass
class RebalancePreset:
    """Named rebalance rule (currently just frequency).

    Later we can extend this to include:
      - keep_mode (rebuild_all vs keep_if_pass_filter)
      - replacement behaviour (match_exits vs full_reoptimize)
      - link to a CriteriaPreset for rebalance-specific scoring.
    """

    name: str
    label: str
    description: str = ""

    frequency: RebalanceFreq = "12M"

    def to_rebalance_config(self) -> RebalanceConfig:
        return RebalanceConfig(
            frequency=self.frequency,
        )


# ----------------------------------------------------------------------
# Registry for saving / loading presets (JSON-backed)
# ----------------------------------------------------------------------


T = TypeVar("T")


def _load_section(
    data: Dict[str, Any],
    key: str,
    cls: Type[T],
) -> Dict[str, T]:
    """Helper to load a section like {"name": {...}} into {name: cls(...)}."""
    section_raw = data.get(key, {})
    out: Dict[str, T] = {}
    for name, payload in section_raw.items():
        if not isinstance(payload, dict):
            continue
        # `name` is also stored as a field for completeness
        payload = {"name": name, **payload}
        out[name] = cls(**payload)
    return out


def _dump_section(objs: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to dump a section of dataclasses into plain dicts."""
    return {name: asdict(obj) for name, obj in objs.items()}


@dataclass
class PresetRegistry:
    """In-memory registry of all presets + JSON persistence helpers.

    This is intended to be used by the Streamlit app:

      - On startup: `PresetRegistry.load(path)` to read saved presets.
      - In a "Preset Manager" UI: modify `registry.universes[...]`, etc.
      - On save: `registry.save(path)` to persist changes.

    The engine itself *does not* depend on this registry; it only consumes
    the concrete Config objects (`UniverseConfig`, `SignalConfig`, etc.).
    """

    universes: Dict[str, UniversePreset] = field(default_factory=dict)
    criteria: Dict[str, CriteriaPreset] = field(default_factory=dict)
    selections: Dict[str, SelectionPreset] = field(default_factory=dict)
    rebalances: Dict[str, RebalancePreset] = field(default_factory=dict)

    # ---- Persistence ---------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "PresetRegistry":
        if not path.exists():
            # Return an empty registry; caller can populate defaults if needed.
            return cls()

        raw = json.loads(path.read_text(encoding="utf-8"))

        universes = _load_section(raw, "universes", UniversePreset)
        criteria = _load_section(raw, "criteria", CriteriaPreset)
        selections = _load_section(raw, "selections", SelectionPreset)
        rebalances = _load_section(raw, "rebalances", RebalancePreset)

        return cls(
            universes=universes,
            criteria=criteria,
            selections=selections,
            rebalances=rebalances,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "universes": _dump_section(self.universes),
            "criteria": _dump_section(self.criteria),
            "selections": _dump_section(self.selections),
            "rebalances": _dump_section(self.rebalances),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    # ---- Convenience helpers ------------------------------------------

    def get_universe_config(self, name: str) -> UniverseConfig:
        return self.universes[name].to_universe_config()

    def get_signal_config(self, name: str) -> SignalConfig:
        return self.criteria[name].to_signal_config()

    def get_selection_config(self, name: str) -> SelectionConfig:
        return self.selections[name].to_selection_config()

    def get_rebalance_config(self, name: str) -> RebalanceConfig:
        return self.rebalances[name].to_rebalance_config()


# ----------------------------------------------------------------------
# Optional: default presets to bootstrap a fresh install
# ----------------------------------------------------------------------


def default_registry() -> PresetRegistry:
    """Return a registry with a few sensible defaults.

    You can call this from the Streamlit app on first run to pre-populate
    the presets JSON (e.g. if the file does not yet exist).
    """
    registry = PresetRegistry()

    # Universe: Equity – Active – Direct (investible)
    registry.universes["equity_active_direct"] = UniversePreset(
        name="equity_active_direct",
        label="Equity – Active – Direct (Investible)",
        description=(
            "Active equity mutual funds, direct plans only, "
            "status=Active, IsPurchaseAvailable=Y, excludes "
            "sector/thematic, ELSS, solution-oriented, liquid/overnight."
        ),
        filters={
            "asset_classes": ["Equity"],
            "direct_only": True,
            "active_only": True,
            # The rest of the detailed filters (categories, etc.) are
            # implemented inside PostgresDataProvider.get_universe()
            # by inspecting `universe_config.filters`.
        },
    )

    # Criteria: 12-month category rank (lower is better)
    registry.criteria["rank_12m_category"] = CriteriaPreset(
        name="rank_12m_category",
        label="12M Category Rank",
        description="Use 12-month performance rank within category; lower is better.",
        direction="asc",
        expression=None,           # use simple column mode
        filter_expression=None,
        rank_scope="category",
    )

    # Selection: Top 15 equal-weight
    registry.selections["top15_equal"] = SelectionPreset(
        name="top15_equal",
        label="Top 15 funds (equal-weight)",
        description="Pick top 15 funds by criteria; require at least 10.",
        mode="top_n",
        top_n=15,
        min_funds=10,
        weight_scheme="equal",
    )

    # Rebalance: Annual
    registry.rebalances["annual"] = RebalancePreset(
        name="annual",
        label="Annual rebalancing",
        description="Rebalance once a year using the chosen criteria.",
        frequency="12M",
    )

    return registry
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
                f"No performance_ranking rows found for as_of={as_of}."
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

        # 2.5) Attach static dimension tables (e.g. scheme_details)
        df = self._attach_selection_dimension_tables(df)

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
    
        def _attach_selection_dimension_tables(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Attach additional static columns (from tables like scheme_details)
            to the per-date selection snapshot used for formulas.

            This lets formulas reference, e.g., scheme_details columns directly,
            while preserving the no-lookahead property (scheme_details is static).
            """
        if df.empty:
            return df

        if "schemecode" not in df.columns:
            return df

        # Unique schemecodes in this snapshot
        codes = sorted({int(x) for x in df["schemecode"].unique() if pd.notna(x)})
        if not codes:
            return df

        # Fetch scheme_details for these codes
        codes_sql = ",".join(str(c) for c in codes)
        sd_query = f"""
        SELECT *
        FROM scheme_details
        WHERE schemecode IN ({codes_sql})
        """
        try:
            sd = run_sql(sd_query)
        except Exception:
            # If this fails, fall back to base snapshot
            return df

        if sd.empty or "schemecode" not in sd.columns:
            return df

        # Deduplicate and keep only genuinely new columns
        sd = sd.drop_duplicates(subset=["schemecode"])
        extra_cols = [c for c in sd.columns if c != "schemecode" and c not in df.columns]
        if not extra_cols:
            return df

        sd_small = sd[["schemecode"] + extra_cols]
        merged = df.merge(sd_small, on="schemecode", how="left")

        return merged
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
from .presets import PresetRegistry, default_registry

__all__ = [
    "BacktestEngine",
    "PostgresDataProvider",
    "run_batch",
    "PresetRegistry",
    "default_registry",
]
```

## engine.py

```python
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import pandas as pd
from dateutil.relativedelta import relativedelta

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
        For single mode: one row per run.
        For rolling_cohort mode: one row per cohort.
    portfolio_periods : pd.DataFrame
        One row per rebalance period (and per cohort, if rolling).
    holdings : pd.DataFrame
        One row per fund per period (and per cohort, if rolling).
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

    Current capabilities
    --------------------
    - mode="single"
        * Single portfolio over a study window, with optional rebalancing.
        * Universe via DataProvider.get_universe().
        * Signals via DataProvider.get_signal_scores() (no lookahead).
        * SelectionConfig: mode="top_n" or "all", equal-weight.
        * NAV walking via DataProvider.get_nav_series().
        * Benchmark (Nifty 500 TRI) via DataProvider.get_benchmark_series().
        * Optional fee drag via BacktestConfig.fees (no tax yet).

        You can optionally provide `rebalance_signal` / `rebalance_selection`
        to use different criteria from the 2nd period onwards.

    - mode="rolling_cohort"
        * Rolling cohorts inside study_window.
        * Each cohort is just a single-mode run with its own [start, end].
        * Cohorts are started every `cohorts.start_frequency` (e.g. "1M")
          and each cohort has a holding horizon of `cohorts.horizon_years`.
        * Summary/periods/holdings include `cohort_no`, `cohort_start`,
          `cohort_end` columns so you can analyse distributions.
    """

    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def run(self, config: BacktestConfig) -> BacktestResult:
        """Run a backtest in the requested mode."""
        mode = getattr(config, "mode", "single")

        if mode == "single":
            return self._run_single(config)
        elif mode == "rolling_cohort":
            return self._run_rolling_cohorts(config)
        else:
            raise ValueError(f"Unsupported BacktestConfig.mode={mode!r}")

    # ------------------------------------------------------------------
    # Single backtest (current behaviour, extended slightly)
    # ------------------------------------------------------------------

    def _run_single(self, config: BacktestConfig) -> BacktestResult:
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

            # Decide which signal/selection to use
            if period_no == 1:
                signal_cfg = config.signal
                selection_cfg = config.selection
            else:
                signal_cfg = config.rebalance_signal or config.signal
                selection_cfg = config.rebalance_selection or config.selection

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
                signal_config=signal_cfg,
            )

            merged = (
                universe_df.merge(scores_df, on="schemecode", how="inner")
                # IMPORTANT: score is already "higher is better" from provider,
                # so we always sort descending.
                .sort_values("score", ascending=False)
            )

            # Selection modes
            if selection_cfg.mode == "top_n":
                top = merged.head(selection_cfg.top_n).copy()
            elif selection_cfg.mode == "all":
                top = merged.copy()
            else:
                raise ValueError(
                    f"Unsupported selection.mode={selection_cfg.mode!r}"
                )

            n = len(top)

            if n < selection_cfg.min_funds:
                raise RuntimeError(
                    f"Only {n} eligible funds on {rebalance_date}, "
                    f"min_funds={selection_cfg.min_funds}"
                )

            # 3) We currently only support equal-weight portfolios
            if selection_cfg.weight_scheme != "equal":
                raise ValueError(
                    f"Unsupported weight_scheme={selection_cfg.weight_scheme!r}"
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

    # ------------------------------------------------------------------
    # Rolling-cohort mode
    # ------------------------------------------------------------------

    def _run_rolling_cohorts(self, config: BacktestConfig) -> BacktestResult:
        """Run rolling cohorts inside the study_window.

        Each cohort:
          - Starts at some date t0 within [window_start, window_end)
          - Has end date t1 = min(t0 + horizon, window_end)
          - Is run as an independent single-mode backtest (same rules)
        """
        if config.cohorts is None:
            raise ValueError(
                "BacktestConfig.mode='rolling_cohort' requires config.cohorts"
            )

        if config.study_window.start is None or config.study_window.end is None:
            raise ValueError("study_window.start and end must be set")

        window_start: date = config.study_window.start
        window_end: date = config.study_window.end

        # 1) Generate cohort start dates using the same calendar generator
        cohort_starts = generate_rebalance_dates(
            start=window_start,
            end=window_end,
            frequency=config.cohorts.start_frequency,
        )

        # 2) Convert horizon_years -> months and build relativedelta
        horizon_months = int(round(config.cohorts.horizon_years * 12))
        if horizon_months <= 0:
            raise ValueError(
                f"Invalid cohorts.horizon_years={config.cohorts.horizon_years}; "
                "must be > 0."
            )
        horizon_delta = relativedelta(months=horizon_months)

        all_summary: List[pd.DataFrame] = []
        all_periods: List[pd.DataFrame] = []
        all_holdings: List[pd.DataFrame] = []

        cohort_no = 0

        for t0 in cohort_starts:
            t1 = t0 + horizon_delta
            if t1 > window_end:
                t1 = window_end
            if t1 <= t0:
                # horizon is too short; skip this cohort
                continue

            cohort_no += 1
            sub_window = config.study_window.__class__(start=t0, end=t1)

            # Build a sub-config that behaves like a single-mode run
            sub_cfg = replace(
                config,
                study_window=sub_window,
                mode="single",
                cohorts=None,
            )

            sub_result = self._run_single(sub_cfg)

            # Annotate with cohort metadata
            s = sub_result.summary.copy()
            s["cohort_no"] = cohort_no
            s["cohort_start"] = t0
            s["cohort_end"] = t1

            p = sub_result.portfolio_periods.copy()
            p["cohort_no"] = cohort_no
            p["cohort_start"] = t0
            p["cohort_end"] = t1

            h = sub_result.holdings.copy()
            if not h.empty:
                h["cohort_no"] = cohort_no
                h["cohort_start"] = t0
                h["cohort_end"] = t1

            all_summary.append(s)
            all_periods.append(p)
            if not h.empty:
                all_holdings.append(h)

        if not all_summary:
            raise RuntimeError(
                "No cohorts generated. Check study_window and cohorts settings."
            )

        summary = pd.concat(all_summary, ignore_index=True)
        portfolio_periods = pd.concat(all_periods, ignore_index=True)
        holdings = (
            pd.concat(all_holdings, ignore_index=True)
            if all_holdings
            else pd.DataFrame()
        )

        # Note: run_id is still the *parent* run name; cohort-specific info
        # lives in the extra columns.
        return BacktestResult(
            run_id=config.name,
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
        "1M": 1,
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

