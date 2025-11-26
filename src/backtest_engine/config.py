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
    name: str                       # e.g. "rank_12m_category"
    source: str = "performance_ranking"
    lookback_months: int = 12
    direction: Literal["asc", "desc"] = "desc"
    rank_scope: Literal["category", "asset_class", "universe"] = "category"
    tie_breaker: Optional[str] = None


@dataclass
class SelectionConfig:
    top_n: int = 15
    min_funds: int = 10
    weight_scheme: Literal["equal"] = "equal"   # more schemes later


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
