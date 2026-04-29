from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch import nn

from marl_environment import make_env


def flatten_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
	# Keep a deterministic order for policy input.
	return np.concatenate(
		[
			np.asarray(obs["spot"], dtype=np.float32).reshape(-1),
			np.asarray(obs["PV"], dtype=np.float32).reshape(-1),
			np.asarray(obs["D"], dtype=np.float32).reshape(-1),
			np.asarray(obs["soc"], dtype=np.float32).reshape(-1),
			np.asarray(obs["hour"], dtype=np.float32).reshape(-1),
		]
	).astype(np.float32)


class SingleAgentAdapter:
	def __init__(self, solution_path: str | None, n_prosumers: int = 14, horizon: int = 24, seed: int = 1234):
		self.env = make_env(solution_path, n_prosumers=n_prosumers, T=horizon, seed=seed)
		self.agent = self.env.agents[0]
		action_space = cast(spaces.Box, self.env.action_spaces[self.agent])
		self.action_shape = tuple(action_space.shape)
		self.action_low = action_space.low.astype(np.float32).reshape(-1)
		self.action_high = action_space.high.astype(np.float32).reshape(-1)
		self.action_dim = int(self.action_low.size)
		self.obs_dim = int(flatten_obs(self._peek_obs()).shape[0])

	def _peek_obs(self) -> dict[str, np.ndarray]:
		self.env.reset()
		return self.env.observe(self.agent)

	def reset(self) -> np.ndarray:
		self.env.reset()
		return flatten_obs(self.env.observe(self.agent))

	def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
		clipped = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), self.action_low, self.action_high).astype(np.float32)
		self.env.agent_selection = self.agent
		self.env.step(clipped.reshape(self.action_shape).tolist())
		reward = float(self.env.rewards[self.agent])
		done = bool(self.env.dones[self.agent])
		info = dict(self.env.infos[self.agent])
		if done:
			next_obs = np.zeros(self.obs_dim, dtype=np.float32)
		else:
			next_obs = flatten_obs(self.env.observe(self.agent))
		return next_obs, reward, done, info


class ReplayBuffer:
	def __init__(self, capacity: int, obs_dim: int, action_dim: int):
		self.capacity = int(capacity)
		self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
		self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
		self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
		self.rewards = np.zeros((capacity, 1), dtype=np.float32)
		self.dones = np.zeros((capacity, 1), dtype=np.float32)
		self.ptr = 0
		self.size = 0

	def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
		i = self.ptr
		self.obs[i] = obs
		self.actions[i] = action
		self.rewards[i] = reward
		self.next_obs[i] = next_obs
		self.dones[i] = float(done)
		self.ptr = (self.ptr + 1) % self.capacity
		self.size = min(self.size + 1, self.capacity)

	def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
		idx = np.random.randint(0, self.size, size=batch_size)
		return (
			torch.as_tensor(self.obs[idx], device=device),
			torch.as_tensor(self.actions[idx], device=device),
			torch.as_tensor(self.rewards[idx], device=device),
			torch.as_tensor(self.next_obs[idx], device=device),
			torch.as_tensor(self.dones[idx], device=device),
		)


class MLP(nn.Module):
	def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(in_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, out_dim),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class Critic(nn.Module):
	def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
		super().__init__()
		self.q1 = MLP(obs_dim + action_dim, 1, hidden_dim)
		self.q2 = MLP(obs_dim + action_dim, 1, hidden_dim)

	def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		x = torch.cat([obs, action], dim=-1)
		return self.q1(x), self.q2(x)


class Actor(nn.Module):
	def __init__(self, obs_dim: int, action_dim: int, action_low: np.ndarray, action_high: np.ndarray, hidden_dim: int = 256):
		super().__init__()
		self.backbone = MLP(obs_dim, hidden_dim, hidden_dim)
		self.mu = nn.Linear(hidden_dim, action_dim)
		self.log_std = nn.Linear(hidden_dim, action_dim)

		scale = (action_high - action_low) / 2.0
		bias = (action_high + action_low) / 2.0
		self.register_buffer("action_scale", torch.as_tensor(scale, dtype=torch.float32))
		self.register_buffer("action_bias", torch.as_tensor(bias, dtype=torch.float32))

	def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		h = self.backbone(obs)
		mu = self.mu(h)
		log_std = self.log_std(h).clamp(-5.0, 2.0)
		return mu, log_std

	def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		mu, log_std = self(obs)
		std = log_std.exp()
		dist = torch.distributions.Normal(mu, std)
		pre_tanh = dist.rsample()
		tanh_action = torch.tanh(pre_tanh)
		action_scale = cast(torch.Tensor, self.action_scale)
		action_bias = cast(torch.Tensor, self.action_bias)
		action = tanh_action * action_scale + action_bias

		# Tanh correction term for reparameterized policy log-prob.
		log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - tanh_action.pow(2) + 1e-6)
		log_prob = log_prob.sum(dim=-1, keepdim=True)

		deterministic = torch.tanh(mu) * action_scale + action_bias
		return action, log_prob, deterministic


@dataclass
class SACConfig:
	episodes: int = 200
	horizon: int = 24
	gamma: float = 0.99
	tau: float = 0.005
	actor_lr: float = 3e-4
	critic_lr: float = 3e-4
	alpha_lr: float = 1e-4
	batch_size: int = 16
	buffer_size: int = 100_000
	warmup_steps: int = 0
	updates_per_step: int = 1
	hidden_dim: int = 256
	seed: int = 42


def set_seed(seed: int) -> None:
	np.random.seed(seed)
	torch.manual_seed(seed)


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
	with torch.no_grad():
		for t, s in zip(target.parameters(), source.parameters()):
			t.data.mul_(1.0 - tau).add_(tau * s.data)


def _scalar_at_t(values: object, t: int, horizon: int) -> float:
	arr = np.asarray(values, dtype=np.float64)
	if arr.ndim == 0:
		return float(arr)
	if arr.ndim == 1:
		idx = min(t, arr.shape[0] - 1)
		return float(arr[idx])
	if arr.shape[-1] == horizon:
		return float(np.take(arr, t, axis=-1).sum())
	if arr.shape[0] == horizon:
		return float(arr[t].sum())
	return float(arr.sum())


def load_optimal_reward(solution_path: Path, horizon: int) -> float:
	if not solution_path.exists():
		return float("nan")
	with solution_path.open("r", encoding="utf-8") as f:
		sol = json.load(f)

	if "objective" in sol:
		return -float(sol["objective"])

	# Fallback reconstruction if objective is not stored.
	total_cost = 0.0
	for t in range(horizon):
		spot = _scalar_at_t(sol.get("spot", 0.0), t, horizon)
		y_im = _scalar_at_t(sol.get("y_im", 0.0), t, horizon)
		y_ex = _scalar_at_t(sol.get("y_ex", 0.0), t, horizon)
		alpha_grid = _scalar_at_t(sol.get("alpha_grid", 75.0), t, horizon)
		beta = _scalar_at_t(sol.get("beta", 0.5), t, horizon)

		p_im = _scalar_at_t(sol.get("p_im", 0.0), t, horizon)
		p_ex = _scalar_at_t(sol.get("p_ex", 0.0), t, horizon)
		p_pen = _scalar_at_t(sol.get("p_pen", 0.0), t, horizon)
		p_plus = _scalar_at_t(sol.get("p_plus", 0.0), t, horizon)
		d_shed = _scalar_at_t(sol.get("d_shed", 0.0), t, horizon)

		market_cost = p_im * (spot + y_im) - p_ex * (spot - y_ex)
		internal_transfer_cost = (1.0 - beta) * y_im * (p_plus - p_im)
		capacity_cost = alpha_grid * p_pen
		load_shedding_cost = 1.25 * alpha_grid * d_shed
		total_cost += market_cost + internal_transfer_cost + capacity_cost + load_shedding_cost

	return -float(total_cost)


def train_sac(config: SACConfig) -> tuple[list[float], float]:
	set_seed(config.seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	env = SingleAgentAdapter(solution_path=None, n_prosumers=14, horizon=config.horizon, seed=config.seed)
	obs_dim = env.obs_dim
	action_dim = env.action_dim

	actor = Actor(obs_dim, action_dim, env.action_low, env.action_high, config.hidden_dim).to(device)
	critic = Critic(obs_dim, action_dim, config.hidden_dim).to(device)
	target_critic = Critic(obs_dim, action_dim, config.hidden_dim).to(device)
	target_critic.load_state_dict(critic.state_dict())

	actor_optim = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
	critic_optim = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)

	log_alpha = torch.tensor(0.0, device=device, requires_grad=True)
	alpha_optim = torch.optim.Adam([log_alpha], lr=config.alpha_lr)
	target_entropy = -float(action_dim)

	rb = ReplayBuffer(config.buffer_size, obs_dim, action_dim)
	episode_returns: list[float] = []
	global_step = 0

	for episode in range(config.episodes):
		obs = env.reset()

		if global_step < config.warmup_steps:
			action = np.random.uniform(env.action_low, env.action_high).astype(np.float32)
		else:
			obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
			with torch.no_grad():
				action_t, _, _ = actor.sample(obs_t)
			action = action_t.squeeze(0).cpu().numpy().astype(np.float32)

		next_obs, reward, done, info = env.step(action)
		rb.add(obs, action, reward, next_obs, done)
		ep_return = reward
		global_step += 1

		if rb.size >= config.batch_size:
			for _ in range(config.updates_per_step):
				o, a, r, no, d = rb.sample(config.batch_size, device)

				with torch.no_grad():
					next_a, next_logp, _ = actor.sample(no)
					tq1, tq2 = target_critic(no, next_a)
					alpha = log_alpha.exp().detach()
					target_v = torch.min(tq1, tq2) - alpha * next_logp
					target_q = r + (1.0 - d) * config.gamma * target_v

				q1, q2 = critic(o, a)
				critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

				critic_optim.zero_grad()
				critic_loss.backward()
				critic_optim.step()

				new_a, logp, _ = actor.sample(o)
				q1_pi, q2_pi = critic(o, new_a)
				q_pi = torch.min(q1_pi, q2_pi)
				actor_loss = (log_alpha.exp().detach() * logp - q_pi).mean()

				actor_optim.zero_grad()
				actor_loss.backward()
				actor_optim.step()

				alpha_loss = -(log_alpha * (logp + target_entropy).detach()).mean()
				alpha_optim.zero_grad()
				alpha_loss.backward()
				alpha_optim.step()

				with torch.no_grad():
					log_alpha.clamp_(-10.0, 2.0)

				_soft_update(target_critic, critic, config.tau)

		mean_action = float(np.mean(action))
		std_action = float(np.std(action))
		episode_returns.append(ep_return)
		print(f"Episode {episode + 1:4d}/{config.episodes}: return={ep_return:10.4f} | mean_action={mean_action:.6f} std={std_action:.6f}")

	optimal = load_optimal_reward(Path("outputs") / "opt_solution.json", config.horizon)
	return episode_returns, optimal


def plot_returns(episode_returns: list[float], optimal_reward: float, out_path: Path) -> None:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	x = np.arange(1, len(episode_returns) + 1)
	y = np.asarray(episode_returns, dtype=np.float64)

	plt.figure(figsize=(10, 5))
	plt.plot(x, y, label="SAC episodic reward", linewidth=1.5)

	if len(y) >= 10:
		win = min(20, len(y))
		ma = np.convolve(y, np.ones(win) / win, mode="valid")
		plt.plot(np.arange(win, len(y) + 1), ma, label=f"Moving average ({win})", linewidth=2.0)

	if np.isfinite(optimal_reward):
		plt.axhline(optimal_reward, color="tab:red", linestyle="--", label="Optimal reward baseline")

	plt.xlabel("Episode")
	plt.ylabel("Episode return")
	plt.title("SAC Training: Episodic Reward vs Optimal Baseline")
	plt.grid(alpha=0.25)
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()


def main() -> None:
	parser = argparse.ArgumentParser(description="Single-agent SAC for dynamic pricing environment")
	parser.add_argument("--episodes", type=int, default=1_000)
	parser.add_argument("--horizon", type=int, default=24)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--warmup-steps", type=int, default=1_000)
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--buffer-size", type=int, default=100_000)
	parser.add_argument("--updates-per-step", type=int, default=1)
	parser.add_argument("--plot-path", type=str, default="Figures/sac_vs_optimal_reward.png")
	args = parser.parse_args()

	config = SACConfig(
		episodes=args.episodes,
		horizon=args.horizon,
		seed=args.seed,
		warmup_steps=args.warmup_steps,
		batch_size=args.batch_size,
		buffer_size=args.buffer_size,
		updates_per_step=args.updates_per_step,
	)

	episode_returns, optimal = train_sac(config)
	plot_path = Path(args.plot_path)
	plot_returns(episode_returns, optimal, plot_path)
	print(f"Saved reward plot to: {plot_path}")


if __name__ == "__main__":
	main()
