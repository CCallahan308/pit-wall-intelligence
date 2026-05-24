"""Ergast API client — historical results & pit stop ground truth.

Used as a validation set against FastF1 outputs and as a fallback for
sessions where FastF1 data is sparse (pre-2018).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pandas as pd

ERGAST_BASE = "https://ergast.com/api/f1"
logger = logging.getLogger(__name__)


def _get(path: str, params: dict[str, Any] | None = None) -> dict:
    url = f"{ERGAST_BASE}/{path}.json"
    params = (params or {}) | {"limit": 1000}
    r = httpx.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def race_results(year: int, rnd: int | None = None) -> pd.DataFrame:
    path = f"{year}" if rnd is None else f"{year}/{rnd}"
    data = _get(f"{path}/results")
    races = data["MRData"]["RaceTable"]["Races"]
    rows = []
    for race in races:
        for res in race["Results"]:
            rows.append(
                {
                    "year": int(race["season"]),
                    "round": int(race["round"]),
                    "race_name": race["raceName"],
                    "circuit": race["Circuit"]["circuitName"],
                    "driver_code": res["Driver"]["code"],
                    "driver_id": res["Driver"]["driverId"],
                    "constructor": res["Constructor"]["name"],
                    "grid": int(res["grid"]),
                    "position": int(res["position"]) if res["position"].isdigit() else None,
                    "points": float(res["points"]),
                    "status": res["status"],
                    "fastest_lap_rank": int(res.get("FastestLap", {}).get("rank", 0) or 0),
                }
            )
    return pd.DataFrame(rows)


def pit_stops(year: int, rnd: int) -> pd.DataFrame:
    data = _get(f"{year}/{rnd}/pitstops")
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return pd.DataFrame()
    rows = []
    for stop in races[0].get("PitStops", []):
        rows.append(
            {
                "year": year,
                "round": rnd,
                "driver_id": stop["driverId"],
                "stop_number": int(stop["stop"]),
                "lap": int(stop["lap"]),
                "time_of_day": stop["time"],
                "duration_s": float(stop["duration"]),
            }
        )
    return pd.DataFrame(rows)


def qualifying(year: int, rnd: int | None = None) -> pd.DataFrame:
    path = f"{year}" if rnd is None else f"{year}/{rnd}"
    data = _get(f"{path}/qualifying")
    races = data["MRData"]["RaceTable"]["Races"]
    rows = []
    for race in races:
        for q in race["QualifyingResults"]:
            rows.append(
                {
                    "year": int(race["season"]),
                    "round": int(race["round"]),
                    "driver_code": q["Driver"]["code"],
                    "driver_id": q["Driver"]["driverId"],
                    "constructor": q["Constructor"]["name"],
                    "position": int(q["position"]),
                    "q1": q.get("Q1"),
                    "q2": q.get("Q2"),
                    "q3": q.get("Q3"),
                }
            )
    return pd.DataFrame(rows)
