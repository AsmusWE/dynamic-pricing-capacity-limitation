from __future__ import annotations

from pathlib import Path
import json
import os
from typing import cast

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv

from consumer_step import solve_consumer_step
from data_loader import load_data as _load_real_data
from network import FeederNetwork, compute_S_base


class MARLEnvironment(AECEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, solution_path=None, n_prosumers=14, T=24, price_clip=(0.0, 10.0), data_dir="Data", seed=1234):
        super().__init__()
        self.agents = ["aggregator"]
        self.possible_agents = self.agents[:]
        self.agent_selection = None
        self._n = int(n_prosumers)
        self.T = int(T)
        self.t = 0
        self.price_low, self.price_high = price_clip

        self.action_spaces = {
            "aggregator": spaces.Box(low=self.price_low, high=self.price_high, shape=(self._n, self.T), dtype=np.float32)
        }
        self.observation_spaces = {
            "aggregator": spaces.Dict(
                {
                    "spot": spaces.Box(low=-1e6, high=1e6, shape=(self.T,), dtype=np.float32),
                    "PV": spaces.Box(low=0.0, high=1e6, shape=(self._n, self.T), dtype=np.float32),
                    "D": spaces.Box(low=0.0, high=1e6, shape=(self._n, self.T), dtype=np.float32),
                    "soc": spaces.Box(low=0.0, high=1e6, shape=(self._n,), dtype=np.float32),
                    "hour": spaces.Box(low=0, high=self.T - 1, shape=(self.T,), dtype=np.int32),
                }
            )
        }

        self.solution: dict[str, object] = {}
        self.data: dict[str, np.ndarray | float] = {}
        self.live_solver = not (solution_path and os.path.exists(solution_path))
        self.soc = np.zeros(self._n, dtype=np.float32)

        repo_root = Path(__file__).resolve().parent
        self.load_data(repo_root / data_dir, self._n, seed)

        # Load feeder network for flow calculations
        self.network = FeederNetwork(repo_root / data_dir, feeder_name="feeder15")
        self.S_base = compute_S_base(
            repo_root / data_dir, "feeder15",
            cast(np.ndarray, self.data.get("D", np.zeros((self._n, self.T)))),
        )

        if solution_path and os.path.exists(solution_path):
            self.load_solution(solution_path)
        else:
            repo_root = Path(__file__).resolve().parent
            self.load_data(repo_root / data_dir, self._n, seed)

        self.dones = {a: False for a in self.agents}
        self.rewards = {a: 0.0 for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def load_solution(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            self.solution = json.load(handle)
        self.live_solver = False
        for key, value in list(self.solution.items()):
            try:
                self.solution[key] = np.array(value)
            except Exception:
                self.solution[key] = value

    def load_data(self, data_root: Path, n_prosumers: int, seed: int = 1234):
        """Load real data matching the Julia pipeline from data processing.jl."""
        self.data = _load_real_data(data_root=data_root, n_prosumers=n_prosumers, seed=seed)

    def _compute_day_reward(self, step_out, network_flows=None):
        """Compute the community-level reward from per-consumer step output.

        Uses the same cost formula as the Julia objective (model.jl, distribution="none"):
            Σₜ [ pⁱᵐ·(spot+yⁱᵐ) - pᵉˣ·(spot-yᵉˣ) + αᵍʳⁱᵈ·pᵖᵉⁿ + (1-β)·yⁱᵐ·(Σp⁺-pⁱᵐ) ]
            + Σᵢₜ VOLL·dˢʰᵉᵈ
        Plus line limit and voltage violation penalties when network_flows is provided.

        The capacity penalty p_pen[t] = max(0, p_import[t] - cap[t]) is computed here
        (consumer_step only solves per-prosumer LPs and does not see the community cap).
        The Julia-specific regularisation terms γ·x̅² + 100·Σω⁺ are excluded; they are
        negligible for optimal solutions and in the RL the price is clipped and
        individual rationality is ensured by the consumer-step decomposition.
        """
        data_source = self.solution if (self.solution and not self.live_solver) else self.data
        spot = np.asarray(data_source.get("spot", np.zeros(self.T)), dtype=np.float64)
        y_im = np.asarray(data_source.get("y_im", np.zeros(self.T)), dtype=np.float64)
        y_ex = np.asarray(data_source.get("y_ex", np.zeros(self.T)), dtype=np.float64)
        alpha_grid = float(data_source.get("alpha_grid", 75.0))
        beta = float(data_source.get("beta", 0.5))
        # cap always comes from data_loader (solution JSON does not store cap)
        cap = np.asarray(self.data.get("cap", np.full(self.T, np.inf)), dtype=np.float64)

        p_im = np.asarray(step_out.get("p_im", np.zeros((self._n, self.T))), dtype=np.float64)
        p_ex = np.asarray(step_out.get("p_ex", np.zeros((self._n, self.T))), dtype=np.float64)
        p_plus = np.asarray(step_out.get("p_plus", np.zeros((self._n, self.T))), dtype=np.float64)
        d_shed = np.asarray(step_out.get("d_shed", np.zeros((self._n, self.T))), dtype=np.float64)

        # Handle both formats:
        #   consumer_step:  p_im/p_ex are per-prosumer (n,T)  → sum to get community totals
        #   optimal solution: p_im/p_ex are community-level (T,) or (T,1)
        if p_im.ndim == 2 and p_im.shape[0] == self._n:
            # Per-prosumer format (from consumer_step): sum across prosumers
            p_import = np.sum(p_im, axis=0)
            p_export = np.sum(p_ex, axis=0)
            internal_flow = np.sum(p_plus, axis=0) - p_import
        else:
            # Community-level format (from optimal solution): use directly
            p_import = p_im.flatten()
            p_export = p_ex.flatten()
            internal_flow = np.sum(p_plus, axis=0) - p_import

        # Capacity penalty: p_pen[t] = max(0, community_import[t] - capacity_limit[t])
        p_pen = np.maximum(0.0, p_import - cap)

        market_cost = p_import * (spot + y_im) - p_export * (spot - y_ex)
        capacity_cost = alpha_grid * p_pen
        internal_transfer_cost = (1.0 - beta) * y_im * internal_flow
        load_shedding_cost = 1.25 * alpha_grid * np.sum(d_shed, axis=0)
        total_cost = float(np.sum(market_cost + capacity_cost + internal_transfer_cost + load_shedding_cost))

        # --- Network line limit and voltage violations ---
        line_violation_cost = 0.0
        voltage_violation_cost = 0.0
        if network_flows is not None:
            lv = np.asarray(network_flows.get("line_violation", 0.0), dtype=np.float64)
            vv = np.asarray(network_flows.get("voltage_violation", 0.0), dtype=np.float64)
            # Penalise line limit violations at the same rate as capacity (alpha_grid = 75 DKK/kW)
            line_violation_cost = float(alpha_grid * np.sum(lv))
            voltage_violation_cost = float(alpha_grid * np.sum(vv))
            total_cost += line_violation_cost + voltage_violation_cost

        breakdown = {
            "market_cost": float(np.sum(market_cost)),
            "capacity_cost": float(np.sum(capacity_cost)),
            "internal_transfer_cost": float(np.sum(internal_transfer_cost)),
            "load_shedding_cost": float(np.sum(load_shedding_cost)),
            "line_violation_cost": line_violation_cost,
            "voltage_violation_cost": voltage_violation_cost,
            "reward": float(-total_cost),
        }
        return -total_cost, breakdown

    def _run_live_consumer_step(self, action):
        if not self.data:
            raise RuntimeError("No generated data available for live solver step")
        payload = {
            "prices": np.nan_to_num(np.asarray(action, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
            "D": np.nan_to_num(np.asarray(cast(np.ndarray, self.data["D"]), dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
            "PV": np.nan_to_num(np.asarray(cast(np.ndarray, self.data["PV"]), dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
            "soc": np.nan_to_num(np.asarray(self.soc, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
            "eta": float(self.data.get("eta", 0.95)),
            "E_max": np.nan_to_num(np.asarray(cast(np.ndarray, self.data.get("E_max", np.ones(self._n) * 5.0)), dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
            "p_ch_max": np.nan_to_num(np.asarray(cast(np.ndarray, self.data.get("p_ch_max", np.ones(self._n) * 2.5)), dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
            "p_dis_max": np.nan_to_num(np.asarray(cast(np.ndarray, self.data.get("p_dis_max", np.ones(self._n) * 2.5)), dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
            "y_im": np.nan_to_num(np.asarray(cast(np.ndarray, self.data.get("y_im", np.zeros(self.T))), dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
            "y_ex": np.nan_to_num(np.asarray(cast(np.ndarray, self.data.get("y_ex", np.zeros(self.T))), dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
        }
        return solve_consumer_step(payload)

    def reset(self, seed=None, options=None):
        self.t = 0
        self.dones = {a: False for a in self.agents}
        self.rewards = {a: 0.0 for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.agent_selection = self.agents[0]
        self.soc = np.zeros(self._n, dtype=np.float32)

    def _get_obs(self, t):
        if (not self.solution) and (not self.data):
            raise RuntimeError("No solution or data loaded")
        if self.data:
            spot = cast(np.ndarray, self.data["spot"])
            PV = cast(np.ndarray, self.data["PV"])
            D = cast(np.ndarray, self.data["D"])
        else:
            spot = np.asarray(self.solution.get("spot", np.zeros(self.T)), dtype=np.float32)
            PV = np.asarray(self.solution.get("PV", np.zeros((self._n, self.T))), dtype=np.float32)
            D = np.asarray(self.solution.get("D", np.zeros((self._n, self.T))), dtype=np.float32)
        return {
            "spot": np.asarray(spot, dtype=np.float32),
            "PV": np.asarray(PV, dtype=np.float32),
            "D": np.asarray(D, dtype=np.float32),
            "soc": np.asarray(self.soc, dtype=np.float32),
            "hour": np.arange(self.T, dtype=np.int32),
        }

    def observe(self, agent):
        return self._get_obs(self.t)

    def step(self, action):
        agent = self.agent_selection
        if self.dones[agent]:
            return
        self.infos[agent]["last_action"] = np.array(action).tolist()

        if self.live_solver:
            out = self._run_live_consumer_step(action)
            if "next_soc" in out:
                self.soc = np.asarray(out["next_soc"], dtype=np.float32).reshape(-1)
            self.infos[agent]["env_out"] = out
            # Compute network flows from consumer outputs
            p_plus_arr = np.asarray(out.get("p_plus", np.zeros((self._n, self.T))), dtype=np.float64)
            p_minus_arr = np.asarray(out.get("p_minus", np.zeros((self._n, self.T))), dtype=np.float64)
            network_flows = self.network.compute_flows(p_plus_arr, p_minus_arr, self.S_base)
            reward, breakdown = self._compute_day_reward(out, network_flows)
            self.rewards[agent] = reward
            self.infos[agent]["reward_breakdown"] = breakdown
        else:
            out = {key: np.asarray(self.solution[key]).tolist() for key in ["p_im", "p_ex", "p_pen", "p_plus", "p_minus", "p_ch", "p_dis", "e", "d_shed", "f_p", "f_q", "next_soc"] if key in self.solution}
            if "next_soc" not in out and "e" in out:
                e_arr = np.asarray(out["e"])
                if e_arr.ndim >= 2:
                    out["next_soc"] = np.asarray(e_arr)[:, -1].tolist()
            self.infos[agent]["env_out"] = out
            # Compute network flows from solution data
            p_plus_sol = np.asarray(out.get("p_plus", np.zeros((self._n, self.T))), dtype=np.float64)
            p_minus_sol = np.asarray(out.get("p_minus", np.zeros((self._n, self.T))), dtype=np.float64)
            network_flows = self.network.compute_flows(p_plus_sol, p_minus_sol, self.S_base)
            reward, breakdown = self._compute_day_reward(out, network_flows)
            self.rewards[agent] = reward
            self.infos[agent]["reward_breakdown"] = breakdown

        self.t = self.T
        self.dones = {a: True for a in self.agents}
        self.agent_selection = None
        next_obs = self._get_obs(0)
        done = True
        info = self.infos[agent]
        return next_obs, reward, done, info

    def close(self):
        pass

    def step_with_action(self, prices):
        self.agent_selection = self.agents[0]
        self.step(prices)
        return self.infos[self.agents[0]].get("env_out", {})


def make_env(solution_path, n_prosumers=14, T=24, seed=1234):
    return MARLEnvironment(solution_path, n_prosumers=n_prosumers, T=T, seed=seed)
