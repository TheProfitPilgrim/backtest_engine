from __future__ import annotations

from dataclasses import asdict
from datetime import date
from typing import Dict, List

import pandas as pd

from .config import BacktestConfig
from .data_provider import DataProvider


class BacktestResult:
    """Container for engine outputs (we'll flesh this out)."""

    def __init__(
        self,
        run_id: str,
        summary: pd.DataFrame,
        portfolio_periods: pd.DataFrame,
        holdings: pd.DataFrame,
    ):
        self.run_id = run_id
        self.summary = summary
        self.portfolio_periods = portfolio_periods
        self.holdings = holdings


class BacktestEngine:
    """Core engine orchestrator.

    For Phase 0 this will only:
      * generate a single rebalance date (window start)
      * form a static equal-weight portfolio
      * NOT YET walk NAVs (we'll add that next)
    """

    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider

    def run(self, config: BacktestConfig) -> BacktestResult:
        # TODO: generate proper run_id based on config hash + timestamp
        run_id = config.name

        # For now: single rebalance at (auto) start date,
        # holding until study_window.end with NO rebalancing.
        if config.study_window.start is None or config.study_window.end is None:
            raise ValueError("For Phase 0, study_window.start and end must be set")

        start_date: date = config.study_window.start
        end_date: date = config.study_window.end
        rebalance_date: date = start_date

        # 1) Get investible universe on that date
        universe_df = self.data_provider.get_universe(
            rebalance_date, config.universe
        )

        # 2) Get signal scores and pick Top N
        scores_df = self.data_provider.get_signal_scores(
            rebalance_date, universe_df["schemecode"].tolist(), config.signal
        )

        merged = (
            universe_df.merge(scores_df, on="schemecode", how="inner")
            .sort_values("score", ascending=(config.signal.direction == "asc"))
        )

        top = merged.head(config.selection.top_n).copy()
        n = len(top)

        if n < config.selection.min_funds:
            raise RuntimeError(
                f"Only {n} eligible funds on {rebalance_date}, "
                f"min_funds={config.selection.min_funds}"
            )

        # 3) Equal-weight portfolio
        top["weight"] = 1.0 / n

        # 4) Pull NAV history for the holding window
        nav_df = self.data_provider.get_nav_series(
            schemecodes=top["schemecode"].tolist(),
            start=start_date,
            end=end_date,
        )

        if nav_df.empty:
            raise RuntimeError(
                "No NAV data returned for selected schemes between "
                f"{start_date} and {end_date}"
            )

        # 5) Compute fund-level total returns (using first and last NAV in window)
        #    For each schemecode, take first nav as entry, last nav as exit.
        nav_sorted = nav_df.sort_values(["schemecode", "date"])
        first_nav = nav_sorted.groupby("schemecode").first()["nav"].rename("nav_start")
        last_nav = nav_sorted.groupby("schemecode").last()["nav"].rename("nav_end")

        nav_stats = pd.concat([first_nav, last_nav], axis=1).reset_index()
        nav_stats["fund_return"] = nav_stats["nav_end"] / nav_stats["nav_start"] - 1.0

        # 6) Merge returns into holdings
        top = top.merge(nav_stats, on="schemecode", how="left")

        # 7) Portfolio-level return and CAGR
        #    Simple weighted sum of fund returns (since all funds invested at start).
        top["contribution"] = top["weight"] * top["fund_return"]
        gross_return = top["contribution"].sum()

        holding_days = (end_date - start_date).days
        if holding_days > 0:
            gross_cagr = (1.0 + gross_return) ** (365.0 / holding_days) - 1.0
        else:
            gross_cagr = float("nan")

        # --- outputs -------------------------------------------------------
        summary = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "name": config.name,
                    "rebalance_date": rebalance_date,
                    "start_date": start_date,
                    "end_date": end_date,
                    "num_funds": n,
                    "gross_return": gross_return,
                    "gross_cagr": gross_cagr,
                    # net_return / net_cagr will come once we add fees/tax
                }
            ]
        )

        portfolio_periods = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "period_no": 1,
                    "rebalance_date": rebalance_date,
                    "start_date": start_date,
                    "end_date": end_date,
                    "num_funds": n,
                    "gross_return": gross_return,
                    "gross_cagr": gross_cagr,
                }
            ]
        )

        holdings = top.assign(
            run_id=run_id,
            period_no=1,
            rebalance_date=rebalance_date,
        )

        return BacktestResult(
            run_id=run_id,
            summary=summary,
            portfolio_periods=portfolio_periods,
            holdings=holdings,
        )
