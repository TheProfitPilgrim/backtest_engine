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