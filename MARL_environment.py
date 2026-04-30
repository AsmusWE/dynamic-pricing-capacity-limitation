from __future__ import annotations

from pathlib import Path

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from data_loader import load_data as _load_real_data


class MARLEnvironment(ParallelEnv):
    """Decentralized energy-community MARL environment.

    Each of the N prosumers is an independent agent that controls its own
    battery (charge / discharge) for one hour at a time.  Agents share the
    day-ahead forecast (spot prices, capacity limits, tariffs) as well as
    their own demand and PV profile, but they do not observe each other's
    state or actions.

    Episode structure: T sequential steps, one per hour of the day.
    """

    metadata = {"render_modes": ["human"], "name": "energy_community_v0"}

    def __init__(
        self,
        n_prosumers: int = 14,
        T: int = 24,
        data_dir: str = "Data",
        seed: int = 1234,
        capacity_bonus: float = 5.0,
    ):
        super().__init__()
        self._n = int(n_prosumers)
        self.T = int(T)
        self.capacity_bonus = float(capacity_bonus)

        self.possible_agents = [f"consumer_{i}" for i in range(self._n)]

        repo_root = Path(__file__).resolve().parent
        data = _load_real_data(data_root=repo_root / data_dir, n_prosumers=self._n, seed=seed)

        self.D: np.ndarray = data["D"].astype(np.float32)           # (n, T) demand kWh/h
        self.PV: np.ndarray = data["PV"].astype(np.float32)         # (n, T) PV kW
        self.spot: np.ndarray = data["spot"].astype(np.float32)     # (T,) DKK/kWh
        self.y_im: np.ndarray = data["y_im"].astype(np.float32)     # (T,) DKK/kWh
        self.y_ex: np.ndarray = data["y_ex"].astype(np.float32)     # (T,) DKK/kWh
        self.cap: np.ndarray = data["cap"].astype(np.float32)       # (T,) kW
        self.E_max: np.ndarray = data["E_max"].astype(np.float32)   # (n,) kWh
        self.p_ch_max: np.ndarray = data["p_ch_max"].astype(np.float32)   # (n,) kW
        self.p_dis_max: np.ndarray = data["p_dis_max"].astype(np.float32) # (n,) kW
        self.eta: float = float(data["eta"])
        self.alpha_grid: float = float(data["alpha_grid"])

        # Per-signal normalization constants (precomputed once)
        self._norm_spot = max(float(self.spot.max()), 1e-8)
        self._norm_cap = max(float(self.cap.max()), 1e-8)
        self._norm_y_im = max(float(self.y_im.max()), 1e-8)
        self._norm_y_ex = max(float(self.y_ex.max()), 1e-8)
        self._norm_D = np.maximum(self.D.max(axis=1), 1e-8)   # (n,)
        self._norm_PV = np.maximum(self.PV.max(axis=1), 1e-8) # (n,)

        # Spaces are identical for all agents — build once and reuse.
        self._obs_space = spaces.Box(
            low=0.0, high=1.0, shape=(6 * self.T + 2,), dtype=np.float32
        )
        self._act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Episode state — populated by reset()
        self.agents: list[str] = []
        self.t: int = 0
        self.soc: np.ndarray = np.zeros(self._n, dtype=np.float32)

    # ------------------------------------------------------------------
    # PettingZoo required interface
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Box:
        # [spot(T), cap(T), y_im(T), y_ex(T), D_i(T), PV_i(T), soc_i(1), t(1)]
        # All values normalized to [0, 1].
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Box:
        # +1 = full charge at p_ch_max, -1 = full discharge at p_dis_max
        return self._act_space

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        self.agents = self.possible_agents[:]
        self.t = 0
        rng = np.random.default_rng(seed)
        self.soc = (rng.uniform(0.0, 1.0, self._n) * self.E_max).astype(np.float32)

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

        # 2. Power balance (DC-side battery formulation, no load shedding)
        #    net_import = D - PV + p_ch/η - η·p_dis
        net_import = self.D[:, t] - self.PV[:, t] + p_ch / self.eta - self.eta * p_dis
        p_plus = np.maximum(0.0, net_import)
        p_minus = np.maximum(0.0, -net_import)

        # 3. Community total import and capacity violation
        total_import = float(p_plus.sum())
        violation = max(0.0, total_import - float(self.cap[t]))

        # 4. Per-agent reward = −individual_cost + collective term
        individual_cost = (
            p_plus * (self.spot[t] + self.y_im[t])
            - p_minus * (self.spot[t] - self.y_ex[t])
        )
        collective = (
            self.capacity_bonus
            if violation == 0.0
            else -self.alpha_grid * violation / self._n
        )
        rewards = {
            f"consumer_{i}": float(-individual_cost[i] + collective)
            for i in range(self._n)
        }

        # 5. SoC update (DC-side: gain p_ch, lose p_dis, no efficiency in SoC equation)
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

    def _get_obs(self, i: int) -> np.ndarray:
        norm_soc = self.soc[i] / self.E_max[i] if self.E_max[i] > 0 else 0.0
        obs = np.concatenate([
            self.spot / self._norm_spot,
            self.cap / self._norm_cap,
            self.y_im / self._norm_y_im,
            self.y_ex / self._norm_y_ex,
            self.D[i] / self._norm_D[i],
            self.PV[i] / self._norm_PV[i],
            np.array([norm_soc], dtype=np.float32),
            np.array([self.t / self.T], dtype=np.float32),
        ]).astype(np.float32)
        # Clip to declared observation-space bounds to guard against float rounding.
        return np.clip(obs, 0.0, 1.0)
