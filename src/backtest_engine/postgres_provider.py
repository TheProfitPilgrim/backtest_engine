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