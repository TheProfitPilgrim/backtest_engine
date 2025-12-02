# src/backtest_engine/formula.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable

import ast

import numpy as np
import pandas as pd
from sqlalchemy import inspect
from sqlalchemy.engine import Engine


# ---------- Metadata: which fields are allowed in formulas ----------


@dataclass
class FieldInfo:
    table: str
    name: str
    type: str  # textual DB type, e.g. "numeric", "double precision", etc.


def load_table_columns(
    engine: Engine,
    table_name: str,
    schema: str = "public",
) -> Dict[str, FieldInfo]:
    """
    Introspect a single table/view and return a mapping of column_name -> FieldInfo.
    """
    inspector = inspect(engine)
    columns = inspector.get_columns(table_name, schema=schema)

    fields: Dict[str, FieldInfo] = {}
    for col in columns:
        col_name = col["name"]
        col_type = str(col["type"])
        fields[col_name] = FieldInfo(
            table=table_name,
            name=col_name,
            type=col_type,
        )
    return fields

# Which tables feed the SELECTION "mart"
SELECTION_MART_TABLES = [
    "performance_ranking",  # time-series ranking/perf snapshot
    "scheme_details",       # static scheme metadata
]


def load_selection_field_registry(
    engine: Engine,
    schema: str = "public",
) -> Dict[str, FieldInfo]:
    """
    Load the registry of *allowed* fields for SELECTION formulas.

    We build a logical "selection mart" by unioning columns from a small,
    curated set of tables (SELECTION_MART_TABLES). For now:
        - performance_ranking
        - scheme_details

    Later you can add more tables here once get_signal_scores joins them.
    """
    fields: Dict[str, FieldInfo] = {}

    for table in SELECTION_MART_TABLES:
        try:
            table_fields = load_table_columns(engine, table_name=table, schema=schema)
        except Exception:
            # If a table is missing or fails introspection, just skip it.
            continue

        for col_name, info in table_fields.items():
            # If multiple tables share a column name, keep the first one we saw.
            # In practice, perf_ranking will be first so its columns win.
            if col_name in fields:
                continue
            fields[col_name] = info

    if not fields:
        raise RuntimeError(
            f"No selection fields found in tables: {', '.join(SELECTION_MART_TABLES)}"
        )

    return fields

# ---------- Safe-ish expression evaluator over a DataFrame ----------


def _zscore(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / x.std(ddof=0)


SAFE_FUNCS: Dict[str, Callable[..., Any]] = {
    "abs": np.abs,
    "log": np.log,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "zscore": _zscore,
    "min": np.minimum,
    "max": np.maximum,
    "pow": np.power,  # allow pow(a, b) for exponent
    # add more if/when needed
}


ALLOWED_AST_NODES: Iterable[type] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num,
    ast.Name,
    ast.Load,
    ast.Call,
    ast.Compare,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.Eq,
    ast.NotEq,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.USub,
)


class FormulaSyntaxError(ValueError):
    """Raised when the formula string has invalid or forbidden syntax."""


class FormulaNameError(ValueError):
    """Raised when the formula references an unknown column or function."""


def _validate_ast(tree: ast.AST, allowed_names: Dict[str, Any]) -> None:
    """
    Walk the parsed AST and ensure:
    - only whitelisted node types are used
    - only whitelisted functions are used
    - all variable names exist in allowed_names
    """
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST_NODES):
            raise FormulaSyntaxError(f"Forbidden expression element: {type(node).__name__}")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise FormulaSyntaxError("Only simple function calls are allowed")
            if node.func.id not in SAFE_FUNCS:
                raise FormulaSyntaxError(f"Function '{node.func.id}' is not allowed")

        if isinstance(node, ast.Name):
            if node.id not in allowed_names and node.id not in SAFE_FUNCS:
                raise FormulaNameError(f"Unknown variable or function: '{node.id}'")


def evaluate_formula_on_df(
    df: pd.DataFrame,
    formula: str,
    allowed_fields: Dict[str, FieldInfo],
) -> pd.Series:
    """
    Evaluate a formula string against a DataFrame.

    - `df` must contain the columns referenced in the formula.
    - `allowed_fields` defines which column names are permitted in formulas;
      for now these are the columns of `performance_ranking`.
    - Returns a Pandas Series (numeric or boolean), indexed like df.

    Supports:
    - arithmetic: +, -, *, /, %, ^, **
    - comparisons: >, >=, <, <=, ==, !=
    - boolean: and, or (also & and | via numpy/pandas)
    - functions: those in SAFE_FUNCS (abs, log, zscore, pow, ...)
    """

    formula = formula.strip()
    if not formula:
        raise ValueError("Formula is empty")

    # Treat ^ as exponent (power), not bitwise XOR.
    # Users can write perf_6m ^ 2 and it becomes perf_6m ** 2.
    formula = formula.replace("^", "**")

    # Build the evaluation context: column_name -> Series
    context: Dict[str, Any] = {}

    for col in df.columns:
        # Only expose columns that are in allowed_fields
        if col in allowed_fields:
            context[col] = df[col]

    # Add allowed functions
    context.update(SAFE_FUNCS)

    # Parse & validate
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as e:
        raise FormulaSyntaxError(f"Invalid formula syntax: {e}") from e

    _validate_ast(tree, context)

    # Compile expression
    compiled = compile(tree, filename="<formula>", mode="eval")

    # Evaluate; Pandas/NumPy will broadcast operations across Series
    result = eval(compiled, {"__builtins__": {}}, context)

    if not isinstance(result, (pd.Series, np.ndarray)):
        # If it’s a scalar, broadcast to a Series
        result = pd.Series(result, index=df.index)

    # Coerce numpy array to Series if needed
    if isinstance(result, np.ndarray):
        result = pd.Series(result, index=df.index)

    return result