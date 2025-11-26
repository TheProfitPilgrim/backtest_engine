"""
Backtest Engine

Thin, config-first backtest engine for Indian MF/PMS strategies.

This package is designed to be driven by YAML/JSON configs and to
separate data access (DataProvider) from backtest logic (Engine).
"""

from .config import BacktestConfig
from .engine import BacktestEngine

__all__ = ["BacktestConfig", "BacktestEngine"]
