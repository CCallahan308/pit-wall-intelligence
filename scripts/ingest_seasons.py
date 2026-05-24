"""Batch-ingest race sessions across multiple seasons.

Skips rounds that are already on disk so the job is resumable. Logs every
attempt with a clear OK / SKIP / FAIL marker so progress is readable in the
background log.

    uv run python scripts/ingest_seasons.py --start 2020 --end 2025
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback

import fastf1

from pitwall.config import RAW_DIR
from pitwall.ingest.fastf1_client import fetch_session, persist, setup_cache

logger = logging.getLogger("ingest_seasons")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
# Quiet down FastF1's own chatter
logging.getLogger("fastf1").setLevel(logging.WARNING)
logging.getLogger("fastf1.core").setLevel(logging.WARNING)


def race_parquet_path(year: int, rnd: int) -> str:
    return str(RAW_DIR / "laps" / f"year={year}" / f"round={rnd:02d}" / "session=R.parquet")


def ingest_one(year: int, rnd: int, name: str) -> str:
    """Return 'OK', 'SKIP', or 'FAIL: <reason>'."""
    path = race_parquet_path(year, rnd)
    try:
        from pathlib import Path

        if Path(path).exists():
            return "SKIP (already on disk)"
        frames = fetch_session(year, rnd, "R")
        if frames is None:
            return "SKIP (no data returned)"
        persist(frames, year, rnd, "R")
        n_laps = len(frames["laps"])
        return f"OK ({n_laps} laps)"
    except Exception as exc:
        return f"FAIL: {type(exc).__name__}: {exc}"


def ingest_season(year: int) -> tuple[int, int, int]:
    """Return (n_ok, n_skip, n_fail)."""
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as exc:
        logger.warning("[%s] cannot fetch schedule: %s", year, exc)
        return 0, 0, 1
    if schedule is None or len(schedule) == 0:
        logger.info("[%s] empty schedule - season probably hasn't started", year)
        return 0, 0, 0

    n_ok = n_skip = n_fail = 0
    for _, ev in schedule.iterrows():
        rnd = int(ev["RoundNumber"])
        name = str(ev["EventName"])
        t0 = time.time()
        status = ingest_one(year, rnd, name)
        elapsed = time.time() - t0
        marker = (
            "OK" if status.startswith("OK") else "SKIP" if status.startswith("SKIP") else "FAIL"
        )
        logger.info(
            "[%s R%02d] %-32s %-8s %4.1fs  %s", year, rnd, name[:32], marker, elapsed, status
        )
        if marker == "OK":
            n_ok += 1
        elif marker == "SKIP":
            n_skip += 1
        else:
            n_fail += 1
    return n_ok, n_skip, n_fail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2020)
    parser.add_argument("--end", type=int, default=2025)
    args = parser.parse_args()

    setup_cache()
    logger.info("Ingesting race sessions for seasons %s-%s", args.start, args.end)

    totals = {"ok": 0, "skip": 0, "fail": 0}
    for year in range(args.start, args.end + 1):
        logger.info("=" * 60)
        logger.info("SEASON %s", year)
        logger.info("=" * 60)
        try:
            ok, skip, fail = ingest_season(year)
        except Exception:
            logger.error("Season %s crashed:\n%s", year, traceback.format_exc())
            totals["fail"] += 1
            continue
        totals["ok"] += ok
        totals["skip"] += skip
        totals["fail"] += fail
        logger.info("[%s] season summary - ok=%d skip=%d fail=%d", year, ok, skip, fail)

    logger.info("=" * 60)
    logger.info("DONE. ok=%d skip=%d fail=%d", totals["ok"], totals["skip"], totals["fail"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
