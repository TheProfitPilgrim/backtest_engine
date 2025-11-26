from __future__ import annotations

from datetime import date
from typing import List

from dateutil.relativedelta import relativedelta


def _months_for_frequency(frequency: str) -> int:
    """Map rebalance frequency code to number of months."""
    if frequency == "NONE":
        return 0
    mapping = {
        "3M": 3,
        "6M": 6,
        "12M": 12,
        "18M": 18,
        "24M": 24,
    }
    if frequency not in mapping:
        raise ValueError(f"Unsupported rebalance frequency: {frequency}")
    return mapping[frequency]


def generate_rebalance_dates(start: date, end: date, frequency: str) -> List[date]:
    """Generate rebalance dates between start and end (inclusive of start).

    Rules:
      - If frequency == "NONE": a single rebalance at `start` (buy & hold).
      - Else: rebalance at `start`, then every N months, stopping
        when the next date would be >= end.

    Examples:
      start=2020-01-01, end=2021-01-01, freq="12M" -> [2020-01-01]
      start=2020-01-01, end=2021-01-01, freq="6M"  -> [2020-01-01, 2020-07-01]
    """
    if start >= end:
        raise ValueError(f"start ({start}) must be before end ({end})")

    months = _months_for_frequency(frequency)
    # No rebalancing: single portfolio from start to end
    if months == 0:
        return [start]

    dates: List[date] = [start]
    current = start

    while True:
        current = current + relativedelta(months=months)
        if current >= end:
            break
        dates.append(current)

    return dates
