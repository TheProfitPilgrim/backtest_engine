from __future__ import annotations

from datetime import date
from typing import List


def generate_rebalance_dates(start: date, end: date, frequency: str) -> List[date]:
    """Generate rebalance dates between start and end.

    TODO: implement properly using pandas.date_range with 'M', '3M', etc.
    For Phase 0 we won't use this yet.
    """
    return [start]
