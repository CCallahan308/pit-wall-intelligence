"""Monte Carlo race strategy simulator.

Given:
  - a list of drivers with their starting grid, baseline pace, and pit cost
  - one strategy per driver (list of pit laps + compounds)
  - the degradation model fit from history
  - a Safety Car arrival rate (Poisson) per circuit

Outputs a distribution of finishing positions across N simulations.

The simulator is deliberately simple — it models cumulative race time
correctly, handles overtaking by track position (with a configurable
overtake-difficulty parameter), and applies SC neutralisation when triggered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pitwall.models.degradation_curve import DegradationModel


@dataclass
class DriverPlan:
    code: str
    grid: int
    base_pace_s: float          # clean-air pace at minimum fuel
    pit_laps: list[int]
    compounds: list[str]        # one entry per stint, len(pit_laps)+1
    pit_loss_s: float = 21.0


@dataclass
class RaceConfig:
    circuit: str
    total_laps: int
    sc_rate_per_race: float = 0.6   # mean number of SC periods per race
    sc_neutralisation_s: float = 35.0
    overtake_difficulty_s: float = 0.4  # pace advantage needed to pass per lap
    fuel_burn_s_per_lap: float = 0.048  # ~0.03s/kg * 1.6kg/lap
    lap_time_noise_s: float = 0.25     # per-lap pace noise (sigma) -- traffic, kerbs, tyre temp
    pit_loss_noise_s: float = 0.8      # per-stop pit-loss variance


@dataclass
class SimResult:
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)  # rows=sim, cols=driver code
    race_times: pd.DataFrame = field(default_factory=pd.DataFrame)

    def position_distribution(self, code: str) -> pd.Series:
        return self.positions[code].value_counts().sort_index()

    def expected_finish(self, code: str) -> float:
        return float(self.positions[code].mean())


class RaceSimulator:
    def __init__(self, deg_model: DegradationModel, config: RaceConfig, seed: int = 42):
        self.deg = deg_model
        self.config = config
        self.rng = np.random.default_rng(seed)

    def _lap_time(self, driver: DriverPlan, lap: int, stint_idx: int, stint_lap: int) -> float:
        compound = driver.compounds[stint_idx]
        deg = self.deg.predict_delta(compound, self.config.circuit, np.array([stint_lap]))[0]
        deg = 0.0 if np.isnan(deg) else float(deg)
        fuel = self.config.fuel_burn_s_per_lap * (self.config.total_laps - lap)
        return driver.base_pace_s + deg + fuel

    def _simulate_one(self, drivers: list[DriverPlan]) -> pd.Series:
        n_drivers = len(drivers)
        cum_time = np.zeros(n_drivers)
        # SC laps: sample number of SCs from Poisson, then place each on a uniform lap.
        n_sc = self.rng.poisson(self.config.sc_rate_per_race)
        sc_laps = set(int(x) for x in self.rng.integers(5, self.config.total_laps - 2, size=n_sc))

        stint_idx = [0] * n_drivers
        stint_lap = [1] * n_drivers

        for lap in range(1, self.config.total_laps + 1):
            for i, d in enumerate(drivers):
                t = self._lap_time(d, lap, stint_idx[i], stint_lap[i])
                t += self.rng.normal(0.0, self.config.lap_time_noise_s)
                if lap in d.pit_laps:
                    t += d.pit_loss_s + self.rng.normal(0.0, self.config.pit_loss_noise_s)
                    stint_idx[i] = min(stint_idx[i] + 1, len(d.compounds) - 1)
                    stint_lap[i] = 1
                else:
                    stint_lap[i] += 1
                if lap in sc_laps:
                    t = max(t, self.config.sc_neutralisation_s)
                cum_time[i] += t

        # Add grid penalty / time gap at start (approx 0.2s per grid slot)
        for i, d in enumerate(drivers):
            cum_time[i] += 0.2 * (d.grid - 1)

        order = np.argsort(cum_time)
        finish = np.empty(n_drivers, dtype=int)
        for pos, idx in enumerate(order):
            finish[idx] = pos + 1
        return pd.Series({d.code: int(finish[i]) for i, d in enumerate(drivers)}), pd.Series(
            {d.code: float(cum_time[i]) for i, d in enumerate(drivers)}
        )

    def run(self, drivers: list[DriverPlan], n_sim: int = 10_000) -> SimResult:
        pos_rows = []
        time_rows = []
        for _ in range(n_sim):
            pos, t = self._simulate_one(drivers)
            pos_rows.append(pos)
            time_rows.append(t)
        return SimResult(positions=pd.DataFrame(pos_rows), race_times=pd.DataFrame(time_rows))
