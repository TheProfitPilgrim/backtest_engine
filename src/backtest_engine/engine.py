from __future__ import annotations

from dataclasses import asdict
from datetime import date
from typing import Dict, List

import pandas as pd

from .config import BacktestConfig
from .data_provider import DataProvider
from .utils.dates import generate_rebalance_dates


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
        """Run a backtest for a single portfolio, possibly with rebalancing.

        Supports:
          - frequency="NONE": single buy-and-hold from start to end
          - frequency in {"3M", "6M", "12M", "18M", "24M"}:
              rebalance at each generated date inside the study window.
        """
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

        period_rows = []
        holdings_frames = []

        equity_gross = 1.0  # compound portfolio returns
        equity_bench = 1.0  # compound benchmark returns

        for idx, rebalance_date in enumerate(rebalance_dates):
            period_no = idx + 1
            period_start = rebalance_date
            period_end = (
                rebalance_dates[idx + 1]
                if idx + 1 < len(rebalance_dates)
                else end_date
            )

            # 1) Get investible universe on the rebalance date
            universe_df = self.data_provider.get_universe(
                rebalance_date, config.universe
            )
            if universe_df.empty:
                raise RuntimeError(f"No eligible universe on {rebalance_date}")

            # 2) Get signal scores and select funds
            scores_df = self.data_provider.get_signal_scores(
                rebalance_date,
                universe_df["schemecode"].tolist(),
                config.signal,
            )

            merged = (
                universe_df.merge(scores_df, on="schemecode", how="inner")
                .sort_values("score", ascending=(config.signal.direction == "asc"))
            )

            sel_cfg = config.selection

            # Selection mode: "top_n" vs "all"
            if sel_cfg.mode == "top_n":
                top = merged.head(sel_cfg.top_n).copy()
            elif sel_cfg.mode == "all":
                # Take all eligible funds (after universe + signal merge)
                top = merged.copy()
            else:
                raise ValueError(f"Unknown selection.mode: {sel_cfg.mode!r}")

            n = len(top)

            if n < sel_cfg.min_funds:
                raise RuntimeError(
                    f"Only {n} eligible funds on {rebalance_date}, "
                    f"min_funds={sel_cfg.min_funds}"
                )

            # 3) Apply weight scheme (currently only equal-weight)
            if sel_cfg.weight_scheme == "equal":
                top["weight"] = 1.0 / n
            else:
                raise ValueError(
                    f"Unknown weight_scheme: {sel_cfg.weight_scheme!r}"
                )

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

            # 6) Portfolio-level return and CAGR for this period
            top["contribution"] = top["weight"] * top["fund_return"]
            gross_return = top["contribution"].sum()

            period_days = (period_end - period_start).days
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

            # Update compounded equity
            equity_gross *= (1.0 + gross_return)
            equity_bench *= (1.0 + bench_return)

            # Store period-level row
            period_rows.append(
                {
                    "run_id": run_id,
                    "period_no": period_no,
                    "rebalance_date": rebalance_date,
                    "start_date": period_start,
                    "end_date": period_end,
                    "num_funds": n,
                    "gross_return": gross_return,
                    "gross_cagr": gross_cagr,
                    "benchmark_return": bench_return,
                    "benchmark_cagr": bench_cagr,
                    "alpha_return": gross_return - bench_return,
                    "alpha_cagr": gross_cagr - bench_cagr,
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
                    period_gross_return=gross_return,
                    period_gross_cagr=gross_cagr,
                    period_benchmark_return=bench_return,
                    period_benchmark_cagr=bench_cagr,
                    period_alpha_return=gross_return - bench_return,
                    period_alpha_cagr=gross_cagr - bench_cagr,
                )
            )

        if not period_rows:
            raise RuntimeError("No valid periods generated for this run.")

        # Backtest-level summary (compounded over all periods)
        total_days = (end_date - start_date).days
        gross_return_total = equity_gross - 1.0
        bench_return_total = equity_bench - 1.0

        if total_days > 0:
            gross_cagr_total = (equity_gross) ** (365.0 / total_days) - 1.0
            bench_cagr_total = (equity_bench) ** (365.0 / total_days) - 1.0
        else:
            gross_cagr_total = float("nan")
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
                    "benchmark_return": bench_return_total,
                    "benchmark_cagr": bench_cagr_total,
                    "alpha_return": gross_return_total - bench_return_total,
                    "alpha_cagr": gross_cagr_total - bench_cagr_total,
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
        )
