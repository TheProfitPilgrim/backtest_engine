from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Global cached engine so we don't recreate it on every query
_engine: Optional[Engine] = None


def get_engine() -> Engine:
    """Create (or return cached) SQLAlchemy engine for the Kairo Postgres DB.

    This assumes you've already started the SSH tunnel in your terminal:
        ./scripts/tunnel.sh start

    And that PGPASSWORD is exported in your shell.
    """
    global _engine
    if _engine is not None:
        return _engine

    PGHOST = os.getenv("PGHOST", "localhost")
    PGPORT = os.getenv("PGPORT", "5433")
    PGDATABASE = os.getenv("PGDATABASE", "kairo_production")
    PGUSER = os.getenv("PGUSER", "sanjay_readonly")
    PGPASSWORD = os.getenv("PGPASSWORD")

    if not PGPASSWORD:
        raise RuntimeError(
            "PGPASSWORD not found. Export it in your terminal before running backtests."
        )

    url = (
        f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}"
        f"@{PGHOST}:{PGPORT}/{PGDATABASE}?sslmode=require"
    )

    _engine = create_engine(url)
    return _engine


def run_sql(query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Run a SQL query against the Postgres DB and return a pandas DataFrame."""
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)
