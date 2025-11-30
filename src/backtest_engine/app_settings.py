from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict

import yaml

# ---------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------

# src/backtest_engine/app_settings.py -> project root is 2 levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# We keep config files under app/config so they live with the app
SETTINGS_PATH = PROJECT_ROOT / "app" / "config" / "app_settings.yaml"
UNIVERSES_PATH = PROJECT_ROOT / "app" / "config" / "universe_presets.yaml"


# ---------------------------------------------------------------------
# Fee & tax settings (app-level defaults)
# ---------------------------------------------------------------------

@dataclass
class FeeSettings:
    apply: bool = True
    annual_bps: float = 100.0  # 1% p.a. drag


@dataclass
class TaxSettings:
    apply: bool = False
    # Stored as percentages to match the Streamlit wizard UI
    stcg_rate: float = 15.0     # 15%
    ltcg_rate: float = 10.0     # 10%
    ltcg_holding_days: int = 365


@dataclass
class AppSettings:
    """Top-level app config.

    Right now this only carries fee & tax defaults, but you can add more
    later (e.g. default study window, default universe, etc.).
    """
    fees: FeeSettings = field(default_factory=FeeSettings)
    tax: TaxSettings = field(default_factory=TaxSettings)


# ---------------------------------------------------------------------
# Load/save app_settings.yaml
# ---------------------------------------------------------------------

def load_app_settings() -> AppSettings:
    """Load app-level settings from YAML, seeding defaults if missing."""
    if not SETTINGS_PATH.exists():
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        settings = AppSettings()
        save_app_settings(settings)
        return settings

    with SETTINGS_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    fees_data = data.get("fees", {}) or {}
    tax_data = data.get("tax", {}) or {}

    fees = FeeSettings(
        apply=bool(fees_data.get("apply", True)),
        annual_bps=float(fees_data.get("annual_bps", 100.0)),
    )
    tax = TaxSettings(
        apply=bool(tax_data.get("apply", False)),
        stcg_rate=float(tax_data.get("stcg_rate", 15.0)),
        ltcg_rate=float(tax_data.get("ltcg_rate", 10.0)),
        ltcg_holding_days=int(tax_data.get("ltcg_holding_days", 365)),
    )

    return AppSettings(fees=fees, tax=tax)


def save_app_settings(settings: AppSettings) -> None:
    """Persist app-level settings to YAML."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SETTINGS_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "fees": asdict(settings.fees),
                "tax": asdict(settings.tax),
            },
            f,
            sort_keys=False,
        )


# ---------------------------------------------------------------------
# Universe presets (universe_presets.yaml)
# ---------------------------------------------------------------------

def load_universe_presets() -> Dict[str, dict]:
    """Return mapping: preset_name -> preset_dict.

    Each preset dict is free-form but we currently use:
      - description: human readable
      - asset_types: list[str]
      - include_categories: list[str]
      - exclude_categories: list[str]
      - only_direct: bool
      - only_active: bool
      - investible_only: bool
      - growth_only: bool

    The engine today only uses `universe_config.preset`, but these
    fields are here so we can later hook them into SQL filters in
    PostgresDataProvider.get_universe().
    """
    if not UNIVERSES_PATH.exists():
        UNIVERSES_PATH.parent.mkdir(parents=True, exist_ok=True)
        presets = {
            "equity_active_direct": {
                "description": (
                    "Equity, Active, Direct, investible, growth – your current "
                    "default equity universe (ex-index/ETF, ex-sector/thematic/etc.)."
                ),
                "asset_types": ["Equity"],
                "include_categories": [],
                "exclude_categories": [],
                "only_direct": True,
                "only_active": True,
                "investible_only": True,
                "growth_only": True,
            }
        }
        save_universe_presets(presets)
        return presets

    with UNIVERSES_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Stored as: {"universes": {name: {...}, ...}}
    return data.get("universes", {})


def save_universe_presets(presets: Dict[str, dict]) -> None:
    """Persist all universe presets to YAML."""
    UNIVERSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with UNIVERSES_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"universes": presets}, f, sort_keys=False)