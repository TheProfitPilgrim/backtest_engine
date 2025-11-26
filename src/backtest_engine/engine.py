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

        # For now: single rebalance at (auto) start date
        # In the next step we'll compute auto_min/auto_max using data provider.
        if config.study_window.start is None or config.study_window.end is None:
            raise ValueError("For Phase 0, study_window.start and end must be set")

        rebalance_date: date = config.study_window.start

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

        # equal-weight portfolio
        top["weight"] = 1.0 / n

        # --- outputs (stub) -------------------------------------------------
        summary = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "name": config.name,
                    "rebalance_date": rebalance_date,
                    "num_funds": n,
                    # we'll add returns later
                }
            ]
        )

        portfolio_periods = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "period_no": 1,
                    "rebalance_date": rebalance_date,
                    "start_date": config.study_window.start,
                    "end_date": config.study_window.end,
                    "num_funds": n,
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
