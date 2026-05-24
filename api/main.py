"""FastAPI inference service for Pit Wall Intelligence.

Two routes:
  GET  /health
  POST /predict_undercut   - undercut success probability for a race state
  POST /simulate_race      - Monte Carlo finishing-position distribution
  GET  /model_info         - metrics from the last train run

Run locally:
    uv run uvicorn api.main:app --reload --port 8000

Container:
    docker build -t pitwall-api -f api/Dockerfile .
    docker run -p 8000:8000 pitwall-api
"""

from __future__ import annotations

import json
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pitwall import __version__
from pitwall.config import PROCESSED_DIR
from pitwall.models.loaders import load_degradation_model, load_undercut_classifier
from pitwall.models.undercut_classifier import FEATURE_COLS
from pitwall.simulation.race_simulator import DriverPlan, RaceConfig, RaceSimulator

app = FastAPI(
    title="Pit Wall Intelligence API",
    version=__version__,
    description="Race strategy & tyre degradation inference for Formula 1.",
)


# ============================================================
# Models cached at import time
# ============================================================

_DEG = load_degradation_model()
_CLF = load_undercut_classifier()
_COMPARISON_PATH = PROCESSED_DIR / "model_comparison.json"


# ============================================================
# Request / response schemas
# ============================================================


class UndercutRequest(BaseModel):
    gap_ahead_s: float = Field(
        ..., ge=-300, le=300, description="Cumulative race-time gap to car ahead, seconds."
    )
    gap_behind_s: float = Field(
        ..., ge=-300, le=300, description="Cumulative race-time gap to car behind, seconds."
    )
    tyre_age: int = Field(..., ge=0, le=80, description="Current tyre age in laps.")
    tyre_age_delta_vs_ahead: float = Field(0.0, ge=-80, le=80)
    tyre_age_delta_vs_behind: float = Field(0.0, ge=-80, le=80)
    compound_idx: int = Field(1, ge=0, le=4, description="0=SOFT 1=MEDIUM 2=HARD 3=INTER 4=WET.")
    current_deg_slope: float = Field(
        0.08, ge=-1.0, le=1.0, description="OLS slope s/lap over recent stint laps."
    )
    expected_fresh_pace_delta: float = Field(0.0, ge=-10, le=10)
    laps_remaining: int = Field(..., ge=1, le=80)
    race_progress_pct: float = Field(..., ge=0.0, le=1.0)
    sc_prob_next_5: float = Field(0.05, ge=0.0, le=1.0)
    pit_loss_circuit_s: float = Field(22.0, ge=10, le=45)
    track_evolution_s_per_lap: float = Field(-0.02, ge=-1.0, le=1.0)


class UndercutResponse(BaseModel):
    prob_gain: float
    base_rate: float
    model_version: str
    inference_ms: float


class DriverPlanRequest(BaseModel):
    code: str = Field(..., min_length=2, max_length=4)
    grid: int = Field(..., ge=1, le=22)
    base_pace_s: float = Field(..., ge=50, le=200)
    pit_laps: list[int] = Field(default_factory=list)
    compounds: list[str] = Field(default_factory=lambda: ["MEDIUM", "HARD"])


class SimulateRequest(BaseModel):
    circuit: str
    total_laps: int = Field(..., ge=30, le=80)
    sc_rate_per_race: float = Field(0.6, ge=0.0, le=3.0)
    pit_loss_s: float = Field(22.0, ge=15, le=40)
    drivers: list[DriverPlanRequest]
    n_sim: int = Field(1000, ge=100, le=5000)


class SimulateResponse(BaseModel):
    expected_finish: dict[str, float]
    p_win: dict[str, float]
    p_podium: dict[str, float]
    n_sim: int
    inference_ms: float


# ============================================================
# Routes
# ============================================================


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "degradation_model_loaded": _DEG is not None,
        "undercut_classifier_loaded": _CLF is not None,
    }


@app.get("/model_info")
def model_info() -> dict:
    """Latest metrics from the most recent train_and_validate run."""
    if not _COMPARISON_PATH.exists():
        raise HTTPException(
            status_code=503, detail="Models not trained yet. Run scripts/train_and_validate.py."
        )
    return json.loads(_COMPARISON_PATH.read_text())


@app.post("/predict_undercut", response_model=UndercutResponse)
def predict_undercut(req: UndercutRequest) -> UndercutResponse:
    if _CLF is None or _CLF.model is None:
        raise HTTPException(status_code=503, detail="Undercut classifier not loaded.")
    t0 = time.perf_counter()
    import pandas as pd

    X = pd.DataFrame([req.model_dump()])[FEATURE_COLS]
    prob = float(_CLF.predict_proba(X)[0])
    base_rate = float(_CLF.metrics.get("base_rate", 0.106))
    return UndercutResponse(
        prob_gain=prob,
        base_rate=base_rate,
        model_version=f"uc-{__version__}",
        inference_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.post("/simulate_race", response_model=SimulateResponse)
def simulate_race(req: SimulateRequest) -> SimulateResponse:
    if _DEG is None:
        raise HTTPException(status_code=503, detail="Degradation model not loaded.")
    if len(req.drivers) < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 drivers to simulate a race.")
    for d in req.drivers:
        if len(d.compounds) != len(d.pit_laps) + 1:
            raise HTTPException(
                status_code=422,
                detail=f"{d.code}: compounds ({len(d.compounds)}) must be len(pit_laps)+1 ({len(d.pit_laps) + 1}).",
            )

    t0 = time.perf_counter()
    plans = [
        DriverPlan(
            code=d.code.upper(),
            grid=d.grid,
            base_pace_s=d.base_pace_s,
            pit_laps=d.pit_laps,
            compounds=[c.upper() for c in d.compounds],
            pit_loss_s=req.pit_loss_s,
        )
        for d in req.drivers
    ]
    cfg = RaceConfig(
        circuit=req.circuit,
        total_laps=req.total_laps,
        sc_rate_per_race=req.sc_rate_per_race,
        overtake_difficulty_s=0.3,
    )
    sim = RaceSimulator(_DEG, cfg, seed=42)
    result = sim.run(plans, n_sim=req.n_sim)

    return SimulateResponse(
        expected_finish={
            code: float(result.expected_finish(code)) for code in result.positions.columns
        },
        p_win={
            code: float((result.positions[code] == 1).mean()) for code in result.positions.columns
        },
        p_podium={
            code: float((result.positions[code] <= 3).mean()) for code in result.positions.columns
        },
        n_sim=req.n_sim,
        inference_ms=round((time.perf_counter() - t0) * 1000, 2),
    )
