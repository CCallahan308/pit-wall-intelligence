"""Measure FastAPI latency. Writes data/processed/api_latency.json.

Boots the in-process app via TestClient (no network), times N requests per
endpoint, reports p50/p95/p99.

This is honest about what it measures: in-process latency on the
benchmarker's machine. Production deploy latency would add network RTT.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to sys.path so `api.main` resolves
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from api.main import app
from pitwall.config import PROCESSED_DIR

UNDERCUT_PAYLOAD = {
    "gap_ahead_s": 1.8,
    "gap_behind_s": 2.5,
    "tyre_age": 18,
    "tyre_age_delta_vs_ahead": -4,
    "tyre_age_delta_vs_behind": 4,
    "compound_idx": 1,
    "current_deg_slope": 0.08,
    "expected_fresh_pace_delta": 1.4,
    "laps_remaining": 32,
    "race_progress_pct": 0.36,
    "sc_prob_next_5": 0.05,
    "pit_loss_circuit_s": 22.5,
    "track_evolution_s_per_lap": -0.02,
}

SIMULATE_PAYLOAD = {
    "circuit": "Italian Grand Prix",
    "total_laps": 53,
    "sc_rate_per_race": 0.6,
    "pit_loss_s": 22.5,
    "n_sim": 500,
    "drivers": [
        {
            "code": "VER",
            "grid": 1,
            "base_pace_s": 82.5,
            "pit_laps": [22],
            "compounds": ["MEDIUM", "HARD"],
        },
        {
            "code": "HAM",
            "grid": 3,
            "base_pace_s": 82.7,
            "pit_laps": [18, 40],
            "compounds": ["SOFT", "MEDIUM", "HARD"],
        },
    ],
}


def time_endpoint(client: TestClient, method: str, path: str, payload: dict | None, n: int) -> dict:
    timings_ms: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = client.get(path) if method == "GET" else client.post(path, json=payload)
        elapsed = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            timings_ms.append(elapsed)
    arr = np.array(timings_ms)
    return {
        "n_ok": len(arr),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "mean_ms": round(float(arr.mean()), 2),
        "max_ms": round(float(arr.max()), 2),
    }


def main() -> None:
    client = TestClient(app)
    print("Benchmarking /predict_undercut (n=100)...")
    pu = time_endpoint(client, "POST", "/predict_undercut", UNDERCUT_PAYLOAD, n=100)
    print(
        f"  p50: {pu['p50_ms']:.1f}ms  |  p95: {pu['p95_ms']:.1f}ms  |  p99: {pu['p99_ms']:.1f}ms"
    )

    print("Benchmarking /simulate_race (n=10, 500 sims each)...")
    sr = time_endpoint(client, "POST", "/simulate_race", SIMULATE_PAYLOAD, n=10)
    print(
        f"  p50: {sr['p50_ms']:.0f}ms  |  p95: {sr['p95_ms']:.0f}ms  |  p99: {sr['p99_ms']:.0f}ms"
    )

    print("Benchmarking /health (n=200)...")
    he = time_endpoint(client, "GET", "/health", None, n=200)
    print(
        f"  p50: {he['p50_ms']:.2f}ms  |  p95: {he['p95_ms']:.2f}ms  |  p99: {he['p99_ms']:.2f}ms"
    )

    result = {
        "predict_undercut": pu,
        "simulate_race_500_sims": sr,
        "health": he,
        "method": "in-process FastAPI TestClient, single-threaded, model loaded at import time",
    }
    (PROCESSED_DIR / "api_latency.json").write_text(json.dumps(result, indent=2))
    print("\nWritten to data/processed/api_latency.json")


if __name__ == "__main__":
    main()
