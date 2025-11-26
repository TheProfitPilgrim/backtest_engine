from __future__ import annotations

from datetime import date
from typing import Iterable, List

import pandas as pd

from .data_provider import DataProvider
from .config import UniverseConfig, SignalConfig
from .db import run_sql


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
        """Return signal scores using the latest available ranking snapshot.

        NOTE: This version *ignores* `as_of` and always uses the most recent
        date in performance_ranking. This is a TEMPORARY hack to verify the
        plumbing end-to-end and WILL introduce lookahead. We'll tighten this
        once we inspect the actual date range in the table.
        """
        col = self._resolve_signal_column(signal_config)

        query = f"""
        WITH max_date AS (
            SELECT MAX(date) AS max_date
            FROM performance_ranking
        ),
        latest AS (
            SELECT
                pr.schemecode,
                pr.{col} AS score
            FROM performance_ranking pr
            CROSS JOIN max_date md
            WHERE pr.date = md.max_date
        )
        SELECT
            schemecode,
            score
        FROM latest;
        """

        df = run_sql(query)
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

        We'll hook this up to your Nifty 500 TRI source later.
        """
        raise NotImplementedError("get_benchmark_series not implemented yet.")
