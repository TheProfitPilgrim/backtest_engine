from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Literal, Optional


RebalanceFreq = Literal["NONE", "3M", "6M", "12M", "18M", "24M"]


@dataclass
class StudyWindow:
    start: Optional[date] = None   # None = auto_min
    end: Optional[date] = None     # None = auto_max


@dataclass
class UniverseConfig:
    # Either use a preset name OR raw filters (we'll implement presets later)
    preset: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SignalConfig:
    """Configuration for how to compute the ranking score.

    Two modes:

    1) Simple column mode (current behaviour):
       - Use `name`, `source`, `lookback_months`, `rank_scope`, etc.
       - Leave `expression` and `filter_expression` as None.
       - Engine will resolve a single column via _resolve_signal_column()
         and use that as `score`.

    2) Expression mode:
       - Set `expression` to a formula using columns in performance_ranking,
         e.g. "0.5 * perf_1y + 0.5 * perf_3y".
       - Optionally set `filter_expression` to a boolean formula to filter funds,
         e.g. "aum_cr > 300 and age_years > 3".
       - `name` becomes just a logical label for this signal.

    direction:
       - "asc": lower score is better (e.g. ranks).
       - "desc": higher score is better (e.g. returns).
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
    frequency: RebalanceFreq = "12M"
    anchor: Literal["month_end"] = "month_end"


@dataclass
class FeeConfig:
    apply: bool = False
    annual_bps: float = 0.0   # 50 = 0.50% p.a.
    apply_frequency: Literal["daily", "monthly", "annual"] = "daily"


@dataclass
class TaxConfig:
    apply: bool = False
    stcg_rate: float = 0.15
    ltcg_rate: float = 0.10
    ltcg_holding_days: int = 365


@dataclass
class BacktestConfig:
    """Top-level config object for a single backtest run.

    This should be directly mappable from your Notion Variant row.
    """

    name: str
    study_window: StudyWindow
    universe: UniverseConfig
    signal: SignalConfig
    selection: SelectionConfig
    rebalance: RebalanceConfig
    fees: FeeConfig = field(default_factory=FeeConfig)
    tax: TaxConfig = field(default_factory=TaxConfig)

    # free-form metadata (links to Notion, code, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)
