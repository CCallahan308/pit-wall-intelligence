"""IO helpers — Parquet glob reads, DuckDB connection."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from pitwall.config import DUCKDB_PATH, RAW_DIR


def read_parquet_glob(table: str, year: int | None = None) -> pd.DataFrame:
    """Read a partitioned parquet table from data/raw/<table>/..."""
    base = RAW_DIR / table
    if not base.exists():
        return pd.DataFrame()
    if year is not None:
        pattern = f"year={year}/round=*/session=*.parquet"
    else:
        pattern = "year=*/round=*/session=*.parquet"
    paths = list(base.glob(pattern))
    if not paths:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)


def duckdb_connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DUCKDB_PATH), read_only=read_only)
