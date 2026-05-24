"""IO helpers - Parquet glob reads, DuckDB queries."""

from __future__ import annotations

import duckdb
import pandas as pd

from pitwall.config import DUCKDB_PATH, RAW_DIR


def read_parquet_glob(table: str, year: int | None = None) -> pd.DataFrame:
    """Read a partitioned parquet table from data/raw/<table>/..."""
    base = RAW_DIR / table
    if not base.exists():
        return pd.DataFrame()
    pattern = (
        f"year={year}/round=*/session=*.parquet"
        if year is not None
        else "year=*/round=*/session=*.parquet"
    )
    paths = list(base.glob(pattern))
    if not paths:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)


def duckdb_connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DUCKDB_PATH), read_only=read_only)


def query(sql: str, params: list | None = None) -> pd.DataFrame:
    """Run a read-only DuckDB query against the warehouse and return a DataFrame.

    Returns an empty DataFrame if the warehouse doesn't exist yet so Streamlit
    pages can render a friendly "no data" message instead of crashing.
    """
    if not DUCKDB_PATH.exists():
        return pd.DataFrame()
    with duckdb_connect(read_only=True) as con:
        return con.execute(sql, params or []).df()


def load_fact_lap(year: int | None = None) -> pd.DataFrame:
    """Return the dbt-built fact_lap table, with the columns the app expects.

    Re-exposes column names in the StudlyCase form the in-memory transforms use
    (StintPosition, LapTimeFuelCorrected, etc.) so plotting code can stay
    agnostic of whether laps came from parquet or DuckDB.
    """
    where = "where year = ?" if year is not None else ""
    params = [year] if year is not None else None
    df = query(
        f"""
        select year, round_num, circuit_name, driver_code, team_name,
               lap_number, stint, stint_position, compound, tyre_life,
               lap_time_s, lap_time_fuel_corrected_s,
               sector1_s, sector2_s, sector3_s,
               position, is_clean_lap
        from fact_lap
        {where}
        """,
        params,
    )
    if df.empty:
        return df
    return df.rename(
        columns={
            "year": "Year",
            "round_num": "Round",
            "circuit_name": "CircuitName",
            "driver_code": "Driver",
            "team_name": "Team",
            "lap_number": "LapNumber",
            "stint": "Stint",
            "stint_position": "StintPosition",
            "compound": "Compound",
            "tyre_life": "TyreLife",
            "lap_time_s": "LapTimeSeconds",
            "lap_time_fuel_corrected_s": "LapTimeFuelCorrected",
            "sector1_s": "Sector1Seconds",
            "sector2_s": "Sector2Seconds",
            "sector3_s": "Sector3Seconds",
            "position": "Position",
            "is_clean_lap": "IsCleanLap",
        }
    )
