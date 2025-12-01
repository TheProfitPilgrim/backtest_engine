# src/backtest_engine/db.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------
# Local dev defaults
# ---------------------------------------------------------------------

# These are safe *for your machine* but obviously shouldn't be kept
# if this repo becomes public. They make life easy for you + your team.
DEFAULT_DB_ENV = {
    "PGHOST": "localhost",
    "PGPORT": "5433",
    "PGDATABASE": "kairo_production",
    "PGUSER": "sanjay_readonly",
    "PGPASSWORD": "Piper112358!",
}

for key, value in DEFAULT_DB_ENV.items():
    if not os.getenv(key):
        os.environ[key] = value


# ---------------------------------------------------------------------
# Engine + helpers
# ---------------------------------------------------------------------

_engine: Optional[Engine] = None


def get_engine() -> Engine:
    """Return a singleton SQLAlchemy engine configured from env vars."""
    global _engine
    if _engine is not None:
        return _engine

    PGHOST = os.getenv("PGHOST", "localhost")
    PGPORT = os.getenv("PGPORT", "5433")
    PGDATABASE = os.getenv("PGDATABASE", "kairo_production")
    PGUSER = os.getenv("PGUSER", "sanjay_readonly")
    PGPASSWORD = os.getenv("PGPASSWORD")

    if not PGPASSWORD:
        # This should rarely happen now because of DEFAULT_DB_ENV,
        # but keep the guardrail for safety.
        raise RuntimeError(
            "PGPASSWORD not found. Export it in your terminal or set DEFAULT_DB_ENV appropriately."
        )

    uri = (
        f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}"
        f"@{PGHOST}:{PGPORT}/{PGDATABASE}?sslmode=require"
    )

    _engine = create_engine(uri)
    return _engine


def run_sql(query: str, params: dict | None = None) -> pd.DataFrame:
    """Run a SQL query and return a DataFrame."""
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


def check_connection() -> bool:
    """Quick sanity check that DB is reachable."""
    try:
        df = run_sql("SELECT now() AT TIME ZONE 'Asia/Kolkata' AS now_ist;")
    except Exception:
        return False
    return not df.empty