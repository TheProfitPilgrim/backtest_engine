# src/backtest_engine/selection.py

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy.engine import Engine

from .formula import (
    evaluate_formula_on_df,
    load_selection_field_registry,
)


@dataclass
class FormulaTopNConfig:
    """
    Config for 'formula_topn' selection.

    Example:
        formula = "0.5 * perf_6m + 0.5 * perf_12m"
        top_n = 10
    """
    formula: str
    top_n: int


def select_topn_by_formula(
    engine: Engine,
    universe_df: pd.DataFrame,
    cfg: FormulaTopNConfig,
    schema: str = "public",
) -> pd.DataFrame:
    """
    Given:
      - a SQLAlchemy Engine,
      - a DataFrame representing the current universe at a given date
        (already joined with performance_ranking),
      - a FormulaTopNConfig containing the formula + N,

    returns:
      - a DataFrame of the selected Top-N schemes, with an extra 'score' column.

    Assumes:
      - universe_df has one row per scheme (schemecode, etc.)
      - universe_df already includes all performance_ranking columns
        you want to use in the formula.
    """
    if universe_df.empty:
        return universe_df

    # 1) Load allowed field metadata (performance_ranking columns)
    field_registry = load_selection_field_registry(engine, schema=schema)

    # 2) Evaluate formula to get a score per row
    score = evaluate_formula_on_df(
        df=universe_df,
        formula=cfg.formula,
        allowed_fields=field_registry,
    )

    # 3) Attach and sort
    df_with_score = universe_df.copy()
    df_with_score["score"] = score

    # Drop rows with NaN score to avoid surprises
    df_with_score = df_with_score.dropna(subset=["score"])

    # 4) Take Top-N by descending score
    selected = df_with_score.sort_values("score", ascending=False).head(cfg.top_n)

    return selected