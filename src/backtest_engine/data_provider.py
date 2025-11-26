from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Iterable, Protocol

import pandas as pd


class DataProvider(ABC):
    """Abstract interface for all data access.

    We'll implement a PostgresDataProvider that talks to your DB via SQLAlchemy.
    """

    @abstractmethod
    def get_universe(
        self,
        as_of: date,
        universe_config,
    ) -> pd.DataFrame:
        """Return universe of schemes valid on as_of date.

        Must include at least: schemecode, scheme_name, classcode, category, etc.
        """

    @abstractmethod
    def get_signal_scores(
        self,
        as_of: date,
        schemecodes: Iterable[int],
        signal_config,
    ) -> pd.DataFrame:
        """Return signal scores for given schemecodes at as_of date.

        Output: columns ['schemecode', 'score'] (and anything else you want).
        """

    @abstractmethod
    def get_nav_series(
        self,
        schemecodes: Iterable[int],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Return NAV history for all schemecodes over [start, end].

        Expected columns: ['date', 'schemecode', 'nav']
        """

    @abstractmethod
    def get_benchmark_series(
        self,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Return benchmark (Nifty 500 TRI) index levels over [start, end]."""
