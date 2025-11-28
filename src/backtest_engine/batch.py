from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Literal, Union

import pandas as pd

from .config import BacktestConfig
from .data_provider import DataProvider
from .engine import BacktestEngine, BacktestResult

SaveLevel = Literal["light", "standard", "full"]

def run_batch(
    configs: Iterable[BacktestConfig],
    data_provider: DataProvider,
    out_dir: Union[str, Path],
    level: SaveLevel = "standard",
) -> pd.DataFrame:
    """Run a batch of backtests and save outputs for each run.

    Parameters
    ----------
    configs : Iterable[BacktestConfig]
        A sequence/list of BacktestConfig objects to run.
    data_provider : DataProvider
        The data provider (e.g. PostgresDataProvider) to use for all runs.
    out_dir : str or Path
        Root directory where outputs for this batch will be written.
        Each run gets its own subdirectory under this.
    level : {"light", "standard", "full"}, default "standard"
        Passed through to BacktestResult.save():
        - "light":    summary only
        - "standard": summary + portfolio_periods
        - "full":     summary + portfolio_periods + holdings (+ config.yaml)

    Returns
    -------
    pd.DataFrame
        Concatenated summary table for all runs, with extra columns
        indicating where outputs were written.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    engine = BacktestEngine(data_provider)

    summary_frames = []

    for cfg in configs:
        run_id = cfg.name or "backtest"
        run_dir = out_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Run the backtest
        result: BacktestResult = engine.run(cfg)

        # Save outputs for this run
        paths: Dict[str, Path] = result.save(run_dir, level=level)

        # Enrich summary with batch metadata
        summary = result.summary.copy()
        summary["output_dir"] = str(run_dir)
        summary["output_files"] = ", ".join(str(p) for p in paths.values())

        summary_frames.append(summary)

    if not summary_frames:
        return pd.DataFrame()

    batch_summary = pd.concat(summary_frames, ignore_index=True)

    # Also save a batch-level summary CSV at the root
    batch_summary_path = out_root / "batch_summary.csv"
    batch_summary.to_csv(batch_summary_path, index=False)

    return batch_summary
