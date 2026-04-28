from pettingzoo.utils.env import AECEnv
from gymnasium import spaces
import numpy as np
import json
import pandas as pd
import os
from typing import cast
from pathlib import Path

from consumer_step import solve_consumer_step


class MARLEnvironment(AECEnv):
	metadata = {"render.modes": ["human"]}

	def _source_scalar(self, source, key, t, default=0.0):
		if not source or key not in source:
			return float(default)
		value = np.asarray(source[key])
		if value.ndim == 0:
			return float(value)
		if value.ndim == 1:
			index = min(int(t), value.shape[0] - 1)
			return float(value[index])
		index = min(int(t), value.shape[-1] - 1)
		return float(np.asarray(np.take(value, index, axis=-1)).sum())

	def _step_sum(self, value):
		if value is None:
			return 0.0
		return float(np.asarray(value, dtype=np.float64).sum())

	def _compute_step_reward(self, step_out, t):
		data_source = self.solution if (self.solution and not self.live_solver) else self.data
		spot = self._source_scalar(data_source, "spot", t, 0.0)
		y_im = self._source_scalar(data_source, "y_im", t, 0.0)
		y_ex = self._source_scalar(data_source, "y_ex", t, 0.0)
		alpha_grid = self._source_scalar(data_source, "alpha_grid", t, 75.0)
		beta = self._source_scalar(data_source, "beta", t, 0.5)
		delta = self._source_scalar(data_source, "Delta", t, 0.0)

		if "p_im" in step_out:
			p_import = self._step_sum(step_out["p_im"])
		elif "p_plus" in step_out:
			p_import = self._step_sum(step_out["p_plus"])
		else:
			p_import = 0.0

		if "p_ex" in step_out:
			p_export = self._step_sum(step_out["p_ex"])
		elif "p_minus" in step_out:
			p_export = self._step_sum(step_out["p_minus"])
		else:
			p_export = 0.0

		if "p_plus" in step_out and "p_im" in step_out:
			internal_flow = self._step_sum(step_out["p_plus"]) - p_import
		else:
			internal_flow = 0.0

		p_pen = self._step_sum(step_out.get("p_pen", 0.0))
		d_shed = self._step_sum(step_out.get("d_shed", 0.0))

		market_cost = p_import * (spot + y_im) - p_export * (spot - y_ex)
		capacity_cost = alpha_grid * p_pen
		internal_transfer_cost = (1.0 - beta) * y_im * internal_flow
		load_shedding_cost = 1.25 * alpha_grid * d_shed
		terminal_cost = 0.0
		if int(t) == int(self.T) - 1:
			xbar = self._source_scalar(data_source, "xbar", t, 0.0)
			omega_plus = self._step_sum(data_source.get("omega_plus", 0.0))
			terminal_cost = delta * (xbar ** 2) + 100.0 * omega_plus

		cost = market_cost + capacity_cost + internal_transfer_cost + load_shedding_cost + terminal_cost
		reward = -cost
		breakdown = {
			"spot": float(spot),
			"y_im": float(y_im),
			"y_ex": float(y_ex),
			"alpha_grid": float(alpha_grid),
			"beta": float(beta),
			"delta": float(delta),
			"p_import": float(p_import),
			"p_export": float(p_export),
			"p_pen": float(p_pen),
			"d_shed": float(d_shed),
			"market_cost": float(market_cost),
			"capacity_cost": float(capacity_cost),
			"internal_transfer_cost": float(internal_transfer_cost),
			"load_shedding_cost": float(load_shedding_cost),
			"terminal_cost": float(terminal_cost),
			"reward": float(reward),
		}
		return reward, breakdown

	def __init__(self, solution_path=None, n_prosumers=14, T=24, price_clip=(0.0, 10.0), data_dir="Data", seed=1234):
		super().__init__()
		self.agents = ["aggregator"]
		self.possible_agents = self.agents[:]
		self.agent_selection = None
		self._n = n_prosumers
		self.T = T
		self.t = 0
		self.price_low, self.price_high = price_clip

		# spaces
		# action: vector of prices for each prosumer
		self.action_spaces = {"aggregator": spaces.Box(low=self.price_low, high=self.price_high, shape=(self._n,), dtype=np.float32)}
		# observation: per-hour data for community (spot price, PV, D) for all prosumers
		obs_shape = (self._n,)
		self.observation_spaces = {"aggregator": spaces.Dict({
			"spot": spaces.Box(low=-1e6, high=1e6, shape=(1,), dtype=np.float32),
			"PV": spaces.Box(low=0.0, high=1e6, shape=obs_shape, dtype=np.float32),
			"D": spaces.Box(low=0.0, high=1e6, shape=obs_shape, dtype=np.float32),
			"soc": spaces.Box(low=0.0, high=1e6, shape=obs_shape, dtype=np.float32),
			"hour": spaces.Box(low=0, high=self.T-1, shape=(1,), dtype=np.int32),
		})}

		# storage for a loaded optimizer solution (optional) and for generated data
		self.solution: dict[str, object] = {}
		# Always initialize `data` as a dict to avoid static type checkers
		# complaining that it may be None. It will be populated by
		# `load_data()` or remain empty if a solution JSON is used instead.
		self.data: dict[str, np.ndarray | float] = {}
		self.live_solver = not (solution_path and os.path.exists(solution_path))
		self.soc = np.zeros(self._n, dtype=np.float32)

		# if a full solution JSON is provided and exists, load it; otherwise generate
		# demand, PV and spot time series from the Data folder to match Julia's
		# preprocessing (seeded RNG to reproduce PV assignment)
		if solution_path and os.path.exists(solution_path):
			self.load_solution(solution_path)
		else:
			repo_root = Path(__file__).resolve().parent
			data_root = repo_root / data_dir
			self.load_data(data_root, n_prosumers, seed)

		# runtime state
		self.dones = {a: False for a in self.agents}
		self.rewards = {a: 0.0 for a in self.agents}
		self.infos = {a: {} for a in self.agents}

	def load_solution(self, path):
		with open(path, "r") as f:
			self.solution = json.load(f)
		self.live_solver = False
		# convert lists to numpy arrays for convenient slicing
		# expected keys: prices (n x T), p_im, p_ex, p_plus, p_minus, ...
		for k, v in list(self.solution.items()):
			try:
				arr = np.array(v)
			except Exception:
				arr = v
			self.solution[k] = arr

	def load_data(self, data_root: Path, n_prosumers: int, seed: int = 1234):
		"""Load CREST demand profiles, PV base, and spot prices and produce
		arrays matching the Julia preprocessing. Results are stored in
		`self.data` as numpy arrays with shapes:
		  - `D`: (n_prosumers, T)
		  - `PV`: (n_prosumers, T)
		  - `spot`: (T,)
		The PV assignment uses the same RNG seed as the Julia code
		(Random.seed!(1234)) and draws `max_cap` from 0..3 inclusive.
		"""
		# ensure path objects
		data_root = Path(data_root)
		T = int(self.T)

		# --- CREST demand parsing ---
		crest_path = data_root / "Load Profile Generator" / "CREST profiles.csv"
		D = np.zeros((n_prosumers, T), dtype=np.float32)
		try:
			# header is a few lines down; use skiprows=3 to reach the header row
			df = pd.read_csv(crest_path, sep=';', decimal=',', skiprows=3, engine='python')
			# find demand column (Julia used var"Net dwelling electricity demand")
			demand_col = None
			for c in df.columns:
				low = str(c).lower()
				if 'net' in low and 'dwell' in low:
					demand_col = c
					break
			if demand_col is None:
				# fallback: look for common short name
				for c in df.columns:
					if 'pnet' in str(c).lower() or 'net' in str(c).lower():
						demand_col = c
						break

			# combine Date + Time into datetime and extract hour
			if 'Date' in df.columns and 'Time' in df.columns:
				dt = pd.to_datetime(
					df['Date'].astype(str) + ' ' + df['Time'].astype(str),
					format='%d/%m/%Y %I.%M.%S %p',
					errors='coerce',
				)
				df['__hour'] = dt.dt.hour
			else:
				# best-effort fallback: try parsing a Time column only
				df['__hour'] = pd.to_datetime(df[df.columns[1]], errors='coerce').dt.hour

			if demand_col is None:
				# give up and leave zeros
				pass
			else:
				for j in range(1, n_prosumers + 1):
					sub = df[df[df.columns[0]] == j]
					if '__hour' not in sub.columns:
						continue
					# sum demand per hour and scale like Julia (./(1000*60))
					hourly = sub.groupby('__hour')[demand_col].sum()
					for h in range(T):
						val = pd.to_numeric(hourly.get(h, 0.0), errors='coerce')
						try:
							safe_val = np.nan_to_num(np.array(val, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6).item()
							safe_val = float(np.clip(safe_val, -1e6, 1e6))
							D[j - 1, h] = safe_val / (1000.0 * 60.0)
						except Exception:
							D[j - 1, h] = 0.0
		except Exception:
			# If anything fails, leave D as zeros
			D = np.zeros((n_prosumers, T), dtype=np.float32)

		# --- PV base and assignment ---
		pv_path = data_root / 'PV.csv'
		PV = np.zeros((n_prosumers, T), dtype=np.float32)
		try:
			pv_df = pd.read_csv(pv_path)
			# parse local_time column if present
			if 'local_time' in pv_df.columns:
				pv_df['local_time'] = pd.to_datetime(pv_df['local_time'], errors='coerce')
				day_mask = (pv_df['local_time'] >= pd.Timestamp(2019, 8, 2)) & (pv_df['local_time'] < pd.Timestamp(2019, 8, 3))
				pv_base = pv_df.loc[day_mask, 'electricity'].astype(float).to_numpy()
			else:
				# fallback: try to find 24 consecutive rows for 2019-08-02
				pv_base = pv_df['electricity'].astype(float).iloc[24:48].to_numpy()
			if pv_base.shape[0] != T:
				# If not exactly T, trim or pad
				if pv_base.shape[0] > T:
					pv_base = pv_base[:T]
				else:
					pv_base = np.pad(pv_base, (0, T - pv_base.shape[0]), constant_values=0.0)

			# use numpy RNG with same seed and draw ints in [0,4) to match Julia 0:3
			rng = np.random.default_rng(seed)
			max_cap = rng.integers(0, 4, size=n_prosumers)
			# pv_base is length T; construct (T, n) then transpose to (n, T)
			pv_mat = np.outer(pv_base, max_cap)
			PV = pv_mat.T.astype(np.float32)
		except Exception:
			PV = np.zeros((n_prosumers, T), dtype=np.float32)

		# --- Spot prices and tariffs ---
		spot = np.zeros(T, dtype=np.float32)
		try:
			price_path = data_root / 'elspotprices.csv'
			price_df = pd.read_csv(price_path)
			if 'HourDK' in price_df.columns:
				price_df['HourDK'] = pd.to_datetime(price_df['HourDK'], errors='coerce')
				day_mask = (price_df['HourDK'] >= pd.Timestamp(2021, 8, 2)) & (price_df['HourDK'] < pd.Timestamp(2021, 8, 3))
				price0208 = price_df.loc[day_mask, 'SpotPriceDKK'].astype(float).to_numpy()
			else:
				price0208 = price_df['SpotPriceDKK'].astype(float).iloc[:T].to_numpy()
			# tariffs
			elafgift = 0.7630
			tso = 0.049 + 0.061 + 0.0022
			prices = (price0208 / 1000.0) + elafgift + tso
			# Julia reversed the vector
			prices = prices[::-1]
			if prices.shape[0] != T:
				if prices.shape[0] > T:
					prices = prices[:T]
				else:
					prices = np.pad(prices, (0, T - prices.shape[0]), constant_values=0.0)
			spot = prices.astype(np.float32)
		except Exception:
			spot = np.zeros(T, dtype=np.float32)

		# store generated data in same orientation used by the environment (n, T)
		D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
		PV = np.nan_to_num(PV, nan=0.0, posinf=0.0, neginf=0.0)
		spot = np.nan_to_num(spot, nan=0.0, posinf=0.0, neginf=0.0)
		self.data = {"D": D, "PV": PV, "spot": spot}
		# also create simple tariffs to mirror Julia
		self.data["y_im"] = np.array([0.2296,0.2296,0.2296,0.2296,0.2296,0.2296,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,2.0666,2.0666,2.0666,2.0666,0.6889,0.6889,0.6889], dtype=np.float32)
		self.data["y_ex"] = np.repeat(0.00375 + 0.000875 + 0.01, T).astype(np.float32)
		self.data["alpha_grid"] = 75.0
		self.data["beta"] = 0.5
		self.data["Delta"] = 0.0
		self.data["eta"] = 0.95
		self.data["E_max"] = np.array([5 * cap for cap in self._infer_max_cap(seed, n_prosumers)], dtype=np.float32)
		self.data["p_ch_max"] = (self.data["E_max"] / 2.0).astype(np.float32)
		self.data["p_dis_max"] = (self.data["E_max"] / 2.0).astype(np.float32)

	def _infer_max_cap(self, seed: int, n_prosumers: int):
		rng = np.random.default_rng(seed)
		return rng.integers(0, 4, size=n_prosumers)

	def _run_live_consumer_step(self, action):
		if not self.data:
			raise RuntimeError("No generated data available for live solver step")

		payload = {
			"prices": np.nan_to_num(np.asarray(action, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
			"hour": int(self.t),
			"D": np.nan_to_num(np.asarray(cast(np.ndarray, self.data["D"])[:, int(self.t)], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
			"PV": np.nan_to_num(np.asarray(cast(np.ndarray, self.data["PV"])[:, int(self.t)], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
			"spot": float(np.asarray(cast(np.ndarray, self.data["spot"]))[int(self.t)]),
			"soc": np.nan_to_num(np.asarray(self.soc, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
			"eta": float(cast(float, self.data["eta"])),
			"E_max": np.nan_to_num(np.asarray(cast(np.ndarray, self.data["E_max"]), dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
			"p_ch_max": np.nan_to_num(np.asarray(cast(np.ndarray, self.data["p_ch_max"]), dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
			"p_dis_max": np.nan_to_num(np.asarray(cast(np.ndarray, self.data["p_dis_max"]), dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
			"y_im": float(np.asarray(cast(np.ndarray, self.data["y_im"]))[int(self.t)]),
			"y_ex": float(np.asarray(cast(np.ndarray, self.data["y_ex"]))[int(self.t)]),
		}

		return solve_consumer_step(payload)

	def reset(self, seed=None, options=None):
		self.t = 0
		self.dones = {a: False for a in self.agents}
		self.rewards = {a: 0.0 for a in self.agents}
		self.infos = {a: {} for a in self.agents}
		self.agent_selection = self.agents[0]
		if self.data and "E_max" in self.data:
			self.soc = np.zeros(self._n, dtype=np.float32)
		else:
			self.soc = np.zeros(self._n, dtype=np.float32)
		# AECEnv.reset should not return observations.
		# Observations are retrieved through observe(agent).

	def _get_obs(self, t):
		if (not self.solution) and (not self.data):
			raise RuntimeError("No solution or data loaded")
		# prefer generated data if present, otherwise fall back to loaded solution
		if self.data:
			spot = cast(np.ndarray, self.data["spot"])
			PV = cast(np.ndarray, self.data["PV"])
			D = cast(np.ndarray, self.data["D"])
		else:
			spot = np.array(self.solution.get("spot", [0.0]))
			PV = np.array(self.solution.get("PV", np.zeros((self._n, self.T))))
			D = np.array(self.solution.get("D", np.zeros((self._n, self.T))))

		spot_t = np.array([spot[int(t)]])
		obs = {"spot": spot_t.astype(np.float32), "PV": PV[:, int(t)].astype(np.float32), "D": D[:, int(t)].astype(np.float32), "soc": np.asarray(self.soc, dtype=np.float32), "hour": np.array([int(t)], dtype=np.int32)}
		if hasattr(self, "soc"):
			obs["soc"] = np.asarray(self.soc, dtype=np.float32)
		return obs

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
				self.soc = np.asarray(out["next_soc"], dtype=np.float32)
			self.infos[agent]["env_out"] = out
			reward, breakdown = self._compute_step_reward(out, self.t)
			self.rewards[agent] = reward
			self.infos[agent]["reward_breakdown"] = breakdown
			self.t += 1
			if self.t >= self.T:
				self.dones = {a: True for a in self.agents}
			self.agent_selection = None
			return

		out = {}
		for key in ["p_im", "p_ex", "p_pen", "p_plus", "p_minus", "p_ch", "p_dis", "e", "d_shed", "f_p", "f_q"]:
			if key in self.solution:
				arr = np.array(self.solution[key])
				if arr.ndim == 0:
					out[key] = arr.item()
				elif arr.ndim == 1:
					out[key] = arr[self.t].tolist() if arr.shape[0] == self.T else arr.tolist()
				else:
					if arr.shape[1] == self.T:
						out[key] = arr[:, self.t].tolist()
					elif arr.shape[0] == self.T:
						out[key] = arr[self.t, :].tolist()
					elif arr.shape[-1] == self.T:
						out[key] = np.take(arr, int(self.t), axis=-1).tolist()
					else:
						out[key] = arr.tolist()

		self.infos[agent]["env_out"] = out
		reward, breakdown = self._compute_step_reward(out, self.t)
		self.rewards[agent] = reward
		self.infos[agent]["reward_breakdown"] = breakdown


		self.t += 1
		if self.t >= self.T:
			self.dones = {a: True for a in self.agents}
		self.agent_selection = None

	def close(self):
		pass

	# convenience method for the test: step with supplied action and return output
	def step_with_action(self, prices):
		# set agent selection
		self.agent_selection = self.agents[0]
		self.step(prices)
		return self.infos[self.agents[0]].get("env_out", {})


def make_env(solution_path, n_prosumers=14, T=24):
	return MARLEnvironment(solution_path, n_prosumers=n_prosumers, T=T)

