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