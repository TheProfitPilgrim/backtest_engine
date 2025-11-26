from __future__ import annotations

from datetime import date
from typing import Iterable, List

import os
import pandas as pd

from .data_provider import DataProvider
from .config import UniverseConfig, SignalConfig
from .db import run_sql

_BENCHMARK_CACHE: dict[str, pd.DataFrame] = {}


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
          - We don't filter by schemecodes here; we intersect with the
            universe later via merge on schemecode.
        """
        col = self._resolve_signal_column(signal_config)

        query = f"""
        WITH latest AS (
            SELECT
                pr.schemecode,
                pr.{col} AS score,
                pr.date,
                ROW_NUMBER() OVER (
                    PARTITION BY pr.schemecode
                    ORDER BY pr.date DESC
                ) AS rn
            FROM performance_ranking pr
            WHERE
                pr.plan_code = 5
                AND pr.date <= :as_of
        )
        SELECT
            schemecode,
            score
        FROM latest
        WHERE rn = 1;
        """

        params = {"as_of": as_of}

        df = run_sql(query, params=params)

        if df.empty:
            raise RuntimeError(
                f"No performance_ranking data found for as_of={as_of}. "
                "Try using a later study_window.start, or check the available "
                "date range in performance_ranking for plan_code=5."
            )

        return df


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