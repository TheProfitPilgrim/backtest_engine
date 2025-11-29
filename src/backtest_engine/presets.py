# src/backtest_engine/presets.py

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Type, TypeVar

import json

from .config import (
    UniverseConfig,
    SignalConfig,
    SelectionConfig,
    RebalanceConfig,
    RebalanceFreq,
)

# ----------------------------------------------------------------------
# Universe presets
# ----------------------------------------------------------------------


@dataclass
class UniversePreset:
    """Named universe definition.

    This is a *UI-level* object that standardises how we describe a universe.
    It maps cleanly onto UniverseConfig (which is what the engine uses).

    Typical filters you might put here (inside `filters`):
      - asset_classes: ["Equity"]
      - include_categories: ["Flexi Cap", "Large Cap", ...]
      - exclude_categories: ["Sectoral / Thematic", "ELSS", ...]
      - direct_only: True
      - active_only: True
      - min_aum_cr: 300.0
    """

    name: str                      # internal key, e.g. "equity_active_direct"
    label: str                     # human label, e.g. "Equity – Active – Direct"
    description: str = ""          # longer explanation (for UI hover/help)
    filters: Dict[str, Any] = field(default_factory=dict)

    def to_universe_config(self) -> UniverseConfig:
        """Convert this preset into a UniverseConfig used by the engine."""
        return UniverseConfig(
            preset=self.name,
            filters=self.filters.copy(),
        )


# ----------------------------------------------------------------------
# Criteria / scoring presets
# ----------------------------------------------------------------------


@dataclass
class CriteriaPreset:
    """Named scoring / filter logic over performance_ranking.

    This is essentially a higher-level wrapper around SignalConfig:
      - `expression` and `filter_expression` are formulas interpreted by
        the formula engine against `performance_ranking` columns.
      - `direction` decides whether higher or lower scores are better.

    Example:
      name="mom_blend_1_3_5y"
      expression="0.4 * perf_1y + 0.4 * perf_3y + 0.2 * perf_5y"
      filter_expression="aum_cr > 300 and age_years >= 3"
      direction="desc"
    """

    name: str
    label: str
    description: str = ""

    # These mirror SignalConfig
    source: str = "performance_ranking"
    lookback_months: int = 12
    direction: Literal["asc", "desc"] = "desc"
    rank_scope: Literal["category", "asset_class", "universe"] = "category"
    tie_breaker: Optional[str] = None

    # Formula parts
    expression: Optional[str] = None
    filter_expression: Optional[str] = None

    def to_signal_config(self) -> SignalConfig:
        """Convert this preset into a SignalConfig for the engine."""
        return SignalConfig(
            name=self.name,
            source=self.source,
            lookback_months=self.lookback_months,
            direction=self.direction,
            rank_scope=self.rank_scope,
            tie_breaker=self.tie_breaker,
            expression=self.expression,
            filter_expression=self.filter_expression,
        )


# ----------------------------------------------------------------------
# Selection presets (top-N / all)
# ----------------------------------------------------------------------


@dataclass
class SelectionPreset:
    """Named selection rule: how many funds to pick, and with what weighting."""

    name: str
    label: str
    description: str = ""

    # These mirror SelectionConfig
    mode: Literal["top_n", "all"] = "top_n"
    top_n: int = 15
    min_funds: int = 10
    weight_scheme: Literal["equal"] = "equal"

    def to_selection_config(self) -> SelectionConfig:
        return SelectionConfig(
            mode=self.mode,
            top_n=self.top_n,
            min_funds=self.min_funds,
            weight_scheme=self.weight_scheme,
        )


# ----------------------------------------------------------------------
# Rebalance presets (time-based for now)
# ----------------------------------------------------------------------


@dataclass
class RebalancePreset:
    """Named rebalance rule (currently just frequency).

    Later we can extend this to include:
      - keep_mode (rebuild_all vs keep_if_pass_filter)
      - replacement behaviour (match_exits vs full_reoptimize)
      - link to a CriteriaPreset for rebalance-specific scoring.
    """

    name: str
    label: str
    description: str = ""

    frequency: RebalanceFreq = "12M"

    def to_rebalance_config(self) -> RebalanceConfig:
        return RebalanceConfig(
            frequency=self.frequency,
        )


# ----------------------------------------------------------------------
# Registry for saving / loading presets (JSON-backed)
# ----------------------------------------------------------------------


T = TypeVar("T")


def _load_section(
    data: Dict[str, Any],
    key: str,
    cls: Type[T],
) -> Dict[str, T]:
    """Helper to load a section like {"name": {...}} into {name: cls(...)}."""
    section_raw = data.get(key, {})
    out: Dict[str, T] = {}
    for name, payload in section_raw.items():
        if not isinstance(payload, dict):
            continue
        # `name` is also stored as a field for completeness
        payload = {"name": name, **payload}
        out[name] = cls(**payload)
    return out


def _dump_section(objs: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to dump a section of dataclasses into plain dicts."""
    return {name: asdict(obj) for name, obj in objs.items()}


@dataclass
class PresetRegistry:
    """In-memory registry of all presets + JSON persistence helpers.

    This is intended to be used by the Streamlit app:

      - On startup: `PresetRegistry.load(path)` to read saved presets.
      - In a "Preset Manager" UI: modify `registry.universes[...]`, etc.
      - On save: `registry.save(path)` to persist changes.

    The engine itself *does not* depend on this registry; it only consumes
    the concrete Config objects (`UniverseConfig`, `SignalConfig`, etc.).
    """

    universes: Dict[str, UniversePreset] = field(default_factory=dict)
    criteria: Dict[str, CriteriaPreset] = field(default_factory=dict)
    selections: Dict[str, SelectionPreset] = field(default_factory=dict)
    rebalances: Dict[str, RebalancePreset] = field(default_factory=dict)

    # ---- Persistence ---------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "PresetRegistry":
        if not path.exists():
            # Return an empty registry; caller can populate defaults if needed.
            return cls()

        raw = json.loads(path.read_text(encoding="utf-8"))

        universes = _load_section(raw, "universes", UniversePreset)
        criteria = _load_section(raw, "criteria", CriteriaPreset)
        selections = _load_section(raw, "selections", SelectionPreset)
        rebalances = _load_section(raw, "rebalances", RebalancePreset)

        return cls(
            universes=universes,
            criteria=criteria,
            selections=selections,
            rebalances=rebalances,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "universes": _dump_section(self.universes),
            "criteria": _dump_section(self.criteria),
            "selections": _dump_section(self.selections),
            "rebalances": _dump_section(self.rebalances),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    # ---- Convenience helpers ------------------------------------------

    def get_universe_config(self, name: str) -> UniverseConfig:
        return self.universes[name].to_universe_config()

    def get_signal_config(self, name: str) -> SignalConfig:
        return self.criteria[name].to_signal_config()

    def get_selection_config(self, name: str) -> SelectionConfig:
        return self.selections[name].to_selection_config()

    def get_rebalance_config(self, name: str) -> RebalanceConfig:
        return self.rebalances[name].to_rebalance_config()


# ----------------------------------------------------------------------
# Optional: default presets to bootstrap a fresh install
# ----------------------------------------------------------------------


def default_registry() -> PresetRegistry:
    """Return a registry with a few sensible defaults.

    You can call this from the Streamlit app on first run to pre-populate
    the presets JSON (e.g. if the file does not yet exist).
    """
    registry = PresetRegistry()

    # Universe: Equity – Active – Direct (investible)
    registry.universes["equity_active_direct"] = UniversePreset(
        name="equity_active_direct",
        label="Equity – Active – Direct (Investible)",
        description=(
            "Active equity mutual funds, direct plans only, "
            "status=Active, IsPurchaseAvailable=Y, excludes "
            "sector/thematic, ELSS, solution-oriented, liquid/overnight."
        ),
        filters={
            "asset_classes": ["Equity"],
            "direct_only": True,
            "active_only": True,
            # The rest of the detailed filters (categories, etc.) are
            # implemented inside PostgresDataProvider.get_universe()
            # by inspecting `universe_config.filters`.
        },
    )

    # Criteria: 12-month category rank (lower is better)
    registry.criteria["rank_12m_category"] = CriteriaPreset(
        name="rank_12m_category",
        label="12M Category Rank",
        description="Use 12-month performance rank within category; lower is better.",
        direction="asc",
        expression=None,           # use simple column mode
        filter_expression=None,
        rank_scope="category",
    )

    # Selection: Top 15 equal-weight
    registry.selections["top15_equal"] = SelectionPreset(
        name="top15_equal",
        label="Top 15 funds (equal-weight)",
        description="Pick top 15 funds by criteria; require at least 10.",
        mode="top_n",
        top_n=15,
        min_funds=10,
        weight_scheme="equal",
    )

    # Rebalance: Annual
    registry.rebalances["annual"] = RebalancePreset(
        name="annual",
        label="Annual rebalancing",
        description="Rebalance once a year using the chosen criteria.",
        frequency="12M",
    )

    return registry