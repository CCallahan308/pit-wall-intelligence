"""FastF1 ingestion: pull lap, telemetry, and weather data per session.

Usage:
    python -m pitwall.ingest.fastf1_client --year 2024 --round 8
    python -m pitwall.ingest.fastf1_client --year 2024 --all
"""

from __future__ import annotations

import argparse
import logging

import fastf1
import pandas as pd

from pitwall.config import CACHE_DIR, RAW_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

SESSION_TYPES = ["FP1", "FP2", "FP3", "Q", "S", "SS", "R"]


def setup_cache() -> None:
    """Point FastF1 at a persistent cache dir — saves hours on re-runs."""
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def fetch_session(year: int, rnd: int, session_type: str) -> dict[str, pd.DataFrame] | None:
    """Pull one session and return a dict of dataframes ready to persist.

    Returns None if the session does not exist (e.g. sprint at a non-sprint round).
    """
    try:
        session = fastf1.get_session(year, rnd, session_type)
        session.load(laps=True, telemetry=False, weather=True, messages=True)
    except Exception as exc:  # FastF1 raises a broad set, normalize here
        logger.warning("skip %s %s %s: %s", year, rnd, session_type, exc)
        return None

    laps = session.laps.reset_index(drop=True).copy()
    laps["Year"] = year
    laps["Round"] = rnd
    laps["SessionType"] = session_type
    laps["CircuitName"] = session.event["EventName"]

    weather = session.weather_data.copy() if session.weather_data is not None else pd.DataFrame()
    if not weather.empty:
        weather["Year"] = year
        weather["Round"] = rnd
        weather["SessionType"] = session_type

    results = (
        session.results.reset_index(drop=True).copy()
        if session.results is not None
        else pd.DataFrame()
    )
    if not results.empty:
        results["Year"] = year
        results["Round"] = rnd
        results["SessionType"] = session_type

    return {"laps": laps, "weather": weather, "results": results}


def persist(frames: dict[str, pd.DataFrame], year: int, rnd: int, session_type: str) -> None:
    """Write each frame to data/raw/<table>/<year>/<round>/<session>.parquet."""
    for table, df in frames.items():
        if df is None or df.empty:
            continue
        path = (
            RAW_DIR
            / table
            / f"year={year}"
            / f"round={rnd:02d}"
            / f"session={session_type}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info("wrote %s rows to %s", len(df), path.relative_to(RAW_DIR.parent))


def ingest_round(year: int, rnd: int, sessions: list[str] | None = None) -> None:
    sessions = sessions or SESSION_TYPES
    for s in sessions:
        frames = fetch_session(year, rnd, s)
        if frames is None:
            continue
        persist(frames, year, rnd, s)


def ingest_season(year: int) -> None:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    for _, ev in schedule.iterrows():
        rnd = int(ev["RoundNumber"])
        logger.info("=== %s round %s — %s ===", year, rnd, ev["EventName"])
        ingest_round(year, rnd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull FastF1 data into data/raw/")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--all", action="store_true", help="Pull every round in the season")
    parser.add_argument("--sessions", nargs="+", default=None, help="Subset of session types")
    args = parser.parse_args()

    setup_cache()

    if args.all:
        ingest_season(args.year)
    elif args.round is not None:
        ingest_round(args.year, args.round, args.sessions)
    else:
        parser.error("Pass either --round N or --all")


if __name__ == "__main__":
    main()
