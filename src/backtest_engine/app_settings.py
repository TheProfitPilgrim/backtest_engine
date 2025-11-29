from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List

import yaml


# Location for app-level config (will be committed to git if you add it)
SETTINGS_PATH = Path("config/app_settings.yaml")
UNIVERSES_PATH = Path("config/universe_presets.yaml")


@dataclass
class FeeSettings:
    apply: bool = True
    annual_bps: float = 100.0  # 1% p.a.


@dataclass
class TaxSettings:
    apply: bool = False
    stcg_rate: float = 0.15    # 15%
    ltcg_rate: float = 0.10    # 10%
    ltcg_holding_days: int = 365


@dataclass
class AppSettings:
    fees: FeeSettings = field(default_factory=FeeSettings)
    tax: TaxSettings = field(default_factory=TaxSettings)


def load_app_settings() -> AppSettings:
    if not SETTINGS_PATH.exists():
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        settings = AppSettings()
        save_app_settings(settings)
        return settings

    with SETTINGS_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Very defensive unpacking
    fees = data.get("fees", {}) or {}
    tax = data.get("tax", {}) or {}

    fee_settings = FeeSettings(
        apply=fees.get("apply", True),
        annual_bps=float(fees.get("annual_bps", 100.0)),
    )
    tax_settings = TaxSettings(
        apply=tax.get("apply", False),
        stcg_rate=float(tax.get("stcg_rate", 0.15)),
        ltcg_rate=float(tax.get("ltcg_rate", 0.10)),
        ltcg_holding_days=int(tax.get("ltcg_holding_days", 365)),
    )

    return AppSettings(fees=fee_settings, tax=tax_settings)


def save_app_settings(settings: AppSettings) -> None:
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


# ---------- Universe presets ----------

def load_universe_presets() -> Dict[str, dict]:
    """Return mapping: preset_name -> preset_dict."""
    if not UNIVERSES_PATH.exists():
        UNIVERSES_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Seed with your current default universe
        presets = {
            "equity_active_direct": {
                "description": "Equity, Active, Direct, investible, growth (current hard-coded universe).",
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

    return data.get("universes", {})


def save_universe_presets(presets: Dict[str, dict]) -> None:
    UNIVERSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with UNIVERSES_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"universes": presets}, f, sort_keys=False)