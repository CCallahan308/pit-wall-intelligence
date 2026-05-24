"""Tests for the FastAPI inference service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_predict_undercut_returns_probability():
    payload = {
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
    r = client.post("/predict_undercut", json=payload)
    if r.status_code == 503:
        pytest.skip("models not trained in this environment")
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["prob_gain"] <= 1.0
    assert body["inference_ms"] > 0
    assert "model_version" in body


def test_predict_undercut_rejects_bad_input():
    # tyre_age out of range
    payload = {
        "gap_ahead_s": 1.8,
        "gap_behind_s": 2.5,
        "tyre_age": 500,  # invalid
        "laps_remaining": 32,
        "race_progress_pct": 0.36,
    }
    r = client.post("/predict_undercut", json=payload)
    assert r.status_code == 422


def test_simulate_race_returns_distribution():
    payload = {
        "circuit": "Italian Grand Prix",
        "total_laps": 53,
        "sc_rate_per_race": 0.6,
        "pit_loss_s": 22.5,
        "n_sim": 100,
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
    r = client.post("/simulate_race", json=payload)
    if r.status_code == 503:
        pytest.skip("models not trained in this environment")
    assert r.status_code == 200
    body = r.json()
    assert set(body["expected_finish"].keys()) == {"VER", "HAM"}
    # Probabilities should be in [0, 1]
    for _code, p in body["p_win"].items():
        assert 0.0 <= p <= 1.0


def test_simulate_race_rejects_mismatched_compounds():
    payload = {
        "circuit": "Italian Grand Prix",
        "total_laps": 53,
        "n_sim": 100,
        "drivers": [
            # 1 pit lap should have 2 compounds, this has 3 -> 422
            {
                "code": "VER",
                "grid": 1,
                "base_pace_s": 82.5,
                "pit_laps": [22],
                "compounds": ["MEDIUM", "HARD", "SOFT"],
            },
            {
                "code": "HAM",
                "grid": 2,
                "base_pace_s": 82.7,
                "pit_laps": [18, 40],
                "compounds": ["SOFT", "MEDIUM", "HARD"],
            },
        ],
    }
    r = client.post("/simulate_race", json=payload)
    assert r.status_code == 422
