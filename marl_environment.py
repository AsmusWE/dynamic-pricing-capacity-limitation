from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from gymnasium import spaces
from pettingzoo import ParallelEnv

from data_loader import (
    load_data as _load_real_data,
    get_available_spot_dates,
    get_available_pv_dates,
    _load_spot_prices,
)


class MARLEnvironment(ParallelEnv):
    """Decentralized energy-community MARL environment.

    Each of the N prosumers is an independent agent that controls its own
    battery (charge / discharge) for one hour at a time.  Agents share the
    day-ahead forecast (spot prices, capacity limits, tariffs) as well as
    their own demand and PV profile, but they do not observe each other's
    state or actions.

    Episode structure: T sequential steps, one per hour of the day.

    When randomize_scenarios=True, each call to reset() samples a fresh
    combination of spot price day, PV day, and demand noise so that agents
    learn a general policy rather than one tuned to a single day.
    Calling reset(options={"reference": True}) always uses the Aug-2
    reference scenario with SoC = 0.5 * E_max for reproducible benchmarking.
    """

    metadata = {"render_modes": ["human"], "name": "energy_community_v0"}

    def __init__(
        self,
        n_prosumers: int = 14,
        T: int = 24,
        data_dir: str = "Data",
        seed: int = 1234,
        capacity_bonus: float = 5.0,
        randomize_scenarios: bool = False,
        demand_noise_std: float = 0.05,
        scenario_seed: int = 0,
        alpha_grid: float | None = None,
        violation_discharge_reward: float = 0.0,
    ):
        super().__init__()
        self._n = int(n_prosumers)
        self.T = int(T)
        self.capacity_bonus = float(capacity_bonus)
        self._randomize = randomize_scenarios
        self._demand_noise_std = float(demand_noise_std)
        self._rng_scenario = np.random.default_rng(scenario_seed)

        self.possible_agents = [f"consumer_{i}" for i in range(self._n)]

        repo_root = Path(__file__).resolve().parent
        self._data_root = repo_root / data_dir

        # --- Load reference scenario (Aug 2) ---
        ref_data = _load_real_data(
            data_root=self._data_root,
            n_prosumers=self._n,
            seed=seed,
            spot_date="2021-08-02",
            pv_date="2019-08-02",
        )
        self._ref_spot: np.ndarray = ref_data["spot"]
        self._ref_D: np.ndarray = ref_data["D"]

        # Battery parameters never change between scenarios
        self.E_max: np.ndarray = ref_data["E_max"].astype(np.float32)
        self.p_ch_max: np.ndarray = ref_data["p_ch_max"].astype(np.float32)
        self.p_dis_max: np.ndarray = ref_data["p_dis_max"].astype(np.float32)
        self.eta: float = float(ref_data["eta"])
        self.alpha_grid: float = float(alpha_grid) if alpha_grid is not None else float(ref_data["alpha_grid"])
        self.violation_discharge_reward: float = float(violation_discharge_reward)
        self.y_im: np.ndarray = ref_data["y_im"].astype(np.float32)
        self.y_ex: np.ndarray = ref_data["y_ex"].astype(np.float32)
        # max_cap needed for PV scaling
        self._max_cap: np.ndarray = (self.E_max / 5.0).astype(np.float32)

        # PV scaling: each prosumer i has PV[i] = max_cap[i] * pv_base
        # We store the base PV profile (24,) and apply scaling in _build_pv()
        self._base_D: np.ndarray = ref_data["D"].astype(np.float32)

        # Build reference PV from ref_data (already scaled per prosumer)
        self._ref_PV: np.ndarray = ref_data["PV"].astype(np.float32)

        # Reference cap derived from reference D, PV, spot
        self._ref_cap: np.ndarray = self._compute_cap(
            self._ref_D, self._ref_PV, self._ref_spot
        )

        # --- Scenario pool (loaded once) ---
        if randomize_scenarios:
            self._spot_dates = get_available_spot_dates(self._data_root)
            # Pre-load all spot vectors
            price_path = self._data_root / "elspotprices.csv"
            self._spot_pool: list[np.ndarray] = [
                _load_spot_prices(price_path, date=d) for d in self._spot_dates
            ]
            # Load full PV file for fast date slicing
            pv_path = self._data_root / "PV.csv"
            self._pv_df = pd.read_csv(pv_path, skiprows=3)
            self._pv_df["local_time"] = pd.to_datetime(self._pv_df["local_time"])
            self._pv_dates = get_available_pv_dates(self._data_root, months=(6, 7, 8, 9))
        else:
            self._spot_dates = []
            self._spot_pool = []
            self._pv_df = None
            self._pv_dates = []

        # --- Fixed spaces ---
        self._obs_space = spaces.Box(
            low=0.0, high=1.0, shape=(6 * self.T + 2,), dtype=np.float32
        )
        self._act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Episode state — populated by reset()
        self.agents: list[str] = []
        self.t: int = 0
        self.soc: np.ndarray = np.zeros(self._n, dtype=np.float32)

        # Tariff normalization constants (tariffs are fixed across scenarios)
        self._norm_y_im = max(float(self.y_im.max()), 1e-8)
        self._norm_y_ex = max(float(self.y_ex.max()), 1e-8)

        # Apply reference scenario as initial state
        self._apply_scenario(self._ref_spot, self._ref_PV, self._ref_D, self._ref_cap)

    # ------------------------------------------------------------------
    # PettingZoo required interface
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Box:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Box:
        return self._act_space

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        self.agents = self.possible_agents[:]
        self.t = 0
        options = options or {}

        if options.get("reference"):
            # Deterministic reference scenario for benchmarking
            self._apply_scenario(self._ref_spot, self._ref_PV, self._ref_D, self._ref_cap)
            # OLD: self.soc = (self.E_max * 0.5).astype(np.float32)
        elif self._randomize:
            spot, PV, D = self._sample_scenario(seed)
            cap = self._compute_cap(D, PV, spot)
            self._apply_scenario(spot, PV, D, cap)
            # OLD: rng = np.random.default_rng(seed)
            # OLD: self.soc = (rng.uniform(0.0, 1.0, self._n) * self.E_max).astype(np.float32)
        else:
            self._apply_scenario(self._ref_spot, self._ref_PV, self._ref_D, self._ref_cap)
            # OLD: rng = np.random.default_rng(seed)
            # OLD: self.soc = (rng.uniform(0.0, 1.0, self._n) * self.E_max).astype(np.float32)

        # Set initial SoC to 0 for all cases
        self.soc = np.zeros(self._n, dtype=np.float32)

        observations = {f"consumer_{i}": self._get_obs(i) for i in range(self._n)}
        infos: dict[str, dict] = {agent: {} for agent in self.agents}
        return observations, infos

    def step(
        self,
        actions: dict[str, np.ndarray],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        t = self.t

        # 1. Battery power from normalized actions
        p_ch = np.zeros(self._n, dtype=np.float32)
        p_dis = np.zeros(self._n, dtype=np.float32)
        for i in range(self._n):
            a = float(np.clip(actions[f"consumer_{i}"][0], -1.0, 1.0))
            if a >= 0.0:
                desired = a * self.p_ch_max[i]
                p_ch[i] = min(desired, self.E_max[i] - self.soc[i], self.p_ch_max[i])
            else:
                desired = -a * self.p_dis_max[i]
                p_dis[i] = min(desired, self.soc[i], self.p_dis_max[i])

        # 2. Power balance per agent (DC-side, no load shedding)
        net_import = self.D[:, t] - self.PV[:, t] + p_ch / self.eta - self.eta * p_dis
        # Per-agent position (for monitoring only)
        p_plus = np.maximum(0.0, net_import)
        p_minus = np.maximum(0.0, -net_import)

        # 3. Community net exchange with grid — internal surpluses offset internal deficits
        community_net = float(net_import.sum())
        p_plus_community = max(0.0, community_net)   # community buys from grid
        p_minus_community = max(0.0, -community_net)  # community sells to grid
        total_import = p_plus_community
        violation = max(0.0, total_import - float(self.cap[t]))

        # 4. Per-agent cost: each agent trades internally at spot; grid tariff is shared equally
        individual_cost = (
            net_import * self.spot[t]
            + (p_plus_community * self.y_im[t] - p_minus_community * self.y_ex[t]) / self._n
        )
        collective = (
            self.capacity_bonus
            if violation <= 1e-4
            else -self.alpha_grid * violation / self._n
        )
        rewards = {
            f"consumer_{i}": float(
                -individual_cost[i]
                + collective
                + (self.violation_discharge_reward * p_dis[i] if violation > 1e-4 else 0.0)
            )
            for i in range(self._n)
        }

        # 5. SoC update (DC-side: gain p_ch, lose p_dis)
        self.soc = np.clip(self.soc + p_ch - p_dis, 0.0, self.E_max).astype(np.float32)

        # 6. Advance time; terminate when all hours are consumed
        self.t += 1
        done = self.t >= self.T
        if done:
            self.agents = []

        terminations = {f"consumer_{i}": done for i in range(self._n)}
        truncations = {f"consumer_{i}": False for i in range(self._n)}

        if done:
            observations = {
                f"consumer_{i}": np.zeros(6 * self.T + 2, dtype=np.float32)
                for i in range(self._n)
            }
        else:
            observations = {f"consumer_{i}": self._get_obs(i) for i in range(self._n)}

        infos = {
            f"consumer_{i}": {
                "p_plus": float(p_plus[i]),
                "p_minus": float(p_minus[i]),
                "p_ch": float(p_ch[i]),
                "p_dis": float(p_dis[i]),
                "individual_cost": float(individual_cost[i]),
                "total_import": total_import,
                "cap": float(self.cap[t]),
                "capacity_violation": violation,
            }
            for i in range(self._n)
        }

        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_scenario(
        self,
        spot: np.ndarray,
        PV: np.ndarray,
        D: np.ndarray,
        cap: np.ndarray,
    ) -> None:
        """Set scenario-dependent attributes and recompute normalization constants."""
        self.spot = spot.astype(np.float32)
        self.PV = PV.astype(np.float32)
        self.D = D.astype(np.float32)
        self.cap = cap.astype(np.float32)
        self._norm_spot = max(float(self.spot.max()), 1e-8)
        self._norm_cap = max(float(self.cap.max()), 1e-8)
        self._norm_D = np.maximum(self.D.max(axis=1), 1e-8)
        self._norm_PV = np.maximum(self.PV.max(axis=1), 1e-8)

    @staticmethod
    def _compute_cap(D: np.ndarray, PV: np.ndarray, spot: np.ndarray) -> np.ndarray:
        """Compute hourly capacity limits from total residual and spot prices."""
        total_residual = float(np.sum(D - PV))
        price_range = float(np.max(spot) - np.min(spot))
        if price_range > 1e-12:
            y = (np.max(spot) - spot) / price_range
        else:
            y = np.ones(len(spot), dtype=np.float64)
        z = y / np.sum(y)
        return (total_residual * z).astype(np.float32)

    def _build_pv(self, pv_base: np.ndarray) -> np.ndarray:
        """Scale a 24-element PV profile to (n, T) using per-prosumer max_cap."""
        PV = np.zeros((self._n, self.T), dtype=np.float32)
        for i in range(self._n):
            PV[i, :] = self._max_cap[i] * pv_base
        return PV

    def _sample_scenario(
        self, seed: int | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random spot/PV/demand scenario."""
        rng = self._rng_scenario if seed is None else np.random.default_rng(seed + 10_000)

        # Random spot price day
        spot_idx = int(rng.integers(len(self._spot_pool)))
        spot = self._spot_pool[spot_idx]

        # Random PV day
        pv_date = self._pv_dates[int(rng.integers(len(self._pv_dates)))]
        ts = pd.Timestamp(pv_date)
        mask = (self._pv_df["local_time"] >= ts) & (
            self._pv_df["local_time"] < ts + pd.Timedelta(days=1)
        )
        pv_base = self._pv_df.loc[mask, "electricity"].values[:self.T].astype(np.float32)
        PV = self._build_pv(pv_base)

        # Demand: base + Gaussian noise
        if self._demand_noise_std > 0.0:
            noise = rng.normal(0.0, self._demand_noise_std, self._base_D.shape)
            D = np.clip(self._base_D + noise, 0.0, None).astype(np.float32)
        else:
            D = self._base_D.copy()

        return spot, PV, D

    def _get_obs(self, i: int) -> np.ndarray:
        """Build observation for agent i with remaining-day features padded to full length.
        
        Structure: [spot_remaining(T), cap_remaining(T), y_im_remaining(T), y_ex_remaining(T),
                    soc(1), D_remaining(T), PV_remaining(T), hours_left(1)]
        
        Each "remaining" component starts at current hour self.t and is padded with 0's
        to maintain constant length T.
        """
        norm_soc = self.soc[i] / self.E_max[i] if self.E_max[i] > 0 else 0.0
        hours_left = self.T - self.t
        
        # Create padded views of remaining features (from current hour t to end of day)
        def pad_remaining(signal, norm_const):
            remaining = signal[self.t:]
            padded = np.zeros(self.T, dtype=np.float32)
            n_remaining = len(remaining)
            padded[:n_remaining] = remaining / norm_const
            return padded
        
        spot_remaining = pad_remaining(self.spot, self._norm_spot)
        cap_remaining = pad_remaining(self.cap, self._norm_cap)
        y_im_remaining = pad_remaining(self.y_im, self._norm_y_im)
        y_ex_remaining = pad_remaining(self.y_ex, self._norm_y_ex)
        D_remaining = pad_remaining(self.D[i], self._norm_D[i])
        PV_remaining = pad_remaining(self.PV[i], self._norm_PV[i])
        
        obs = np.concatenate([
            spot_remaining,
            cap_remaining,
            y_im_remaining,
            y_ex_remaining,
            np.array([norm_soc], dtype=np.float32),
            D_remaining,
            PV_remaining,
            np.array([hours_left / self.T], dtype=np.float32),
        ]).astype(np.float32)
        
        return np.clip(obs, 0.0, 1.0)
