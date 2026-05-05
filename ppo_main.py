from __future__ import annotations

import argparse
import copy
import dataclasses
from dataclasses import dataclass
from pathlib import Path
import time
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    import wandb  # type: ignore[import]
except ImportError:
    wandb = None  # type: ignore[assignment]

from marl_environment import MARLEnvironment


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkRollout:
    episode: int
    policy_reward: float
    policy_import: np.ndarray
    total_charging_reward: float
    policy_p_ch: np.ndarray
    policy_p_dis: np.ndarray
    lp_reward: float | None
    lp_import: np.ndarray | None
    lp_p_ch: np.ndarray | None
    lp_p_dis: np.ndarray | None


# ---------------------------------------------------------------------------
# Rollout buffer (trajectory storage for PPO)
# ---------------------------------------------------------------------------


class RolloutBuffer:
    def __init__(self, horizon: int, n_battery: int, obs_dim: int, action_dim: int):
        self.horizon = int(horizon)
        self.n_battery = int(n_battery)
        self.obs = np.zeros((horizon, n_battery, obs_dim), dtype=np.float32)
        self.actions = np.zeros((horizon, n_battery, action_dim), dtype=np.float32)
        self.rewards = np.zeros((horizon, n_battery), dtype=np.float32)
        self.values = np.zeros((horizon, n_battery), dtype=np.float32)
        self.log_probs = np.zeros((horizon, n_battery), dtype=np.float32)
        self.dones = np.zeros((horizon, n_battery), dtype=np.float32)
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.values[self.ptr] = values
        self.log_probs[self.ptr] = log_probs
        self.dones[self.ptr] = dones
        self.ptr += 1

    def compute_returns_and_advantages(
        self, gamma: float, gae_lambda: float, final_values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute discounted returns and GAE advantages."""
        advantages = np.zeros_like(self.rewards)
        gae = np.zeros(self.n_battery, dtype=np.float32)
        next_values = final_values

        for t in reversed(range(self.horizon)):
            if t == self.horizon - 1:
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + self.values
        return returns, advantages

    def get_batch(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample a minibatch from collected trajectories."""
        n_samples = self.horizon * self.n_battery
        indices = np.random.choice(n_samples, size=min(batch_size, n_samples), replace=False)

        traj_idx = indices // self.n_battery
        agent_idx = indices % self.n_battery

        return (
            torch.as_tensor(self.obs[traj_idx, agent_idx], device=device, dtype=torch.float32),
            torch.as_tensor(self.actions[traj_idx, agent_idx], device=device, dtype=torch.float32),
            torch.as_tensor(self.log_probs[traj_idx, agent_idx], device=device, dtype=torch.float32),
            torch.as_tensor(self.returns[traj_idx, agent_idx], device=device, dtype=torch.float32),
            torch.as_tensor(self.advantages[traj_idx, agent_idx], device=device, dtype=torch.float32),
        )

    def reset(self) -> None:
        self.ptr = 0


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


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


class PPOActor(nn.Module):
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

        log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - tanh_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        deterministic = torch.tanh(mu) * action_scale + action_bias
        return action, log_prob, deterministic


class PPOCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = MLP(obs_dim, 1, hidden_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ---------------------------------------------------------------------------
# PPO configuration
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    n_prosumers: int = 14
    episodes: int = 200
    horizon: int = 24
    capacity_bonus: float = 5.0
    data_dir: str = "Data"
    gamma: float = 0.999
    gae_lambda: float = 0.95
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    batch_size: int = 256
    n_updates: int = 10
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.001
    hidden_dim: int = 256
    seed: int = 42
    randomize_scenarios: bool = True
    demand_noise_std: float = 0.05
    benchmark_interval: int = 50
    alpha_grid: float = 75.0
    violation_discharge_reward: float = 0.0
    use_wandb: bool = False
    wandb_project: str = "marl-energy"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _load_lp_actions(data_dir: str) -> np.ndarray | None:
    """Load pre-computed LP optimal actions from Data/optimal_actions.csv.

    Returns (n_prosumers, T) float32 array indexed [prosumer, hour], or None
    if the file does not exist.
    """
    import pandas as pd
    path = Path(data_dir) / "optimal_actions.csv"
    if not path.exists():
        print(
            f"[Benchmark] optimal_actions.csv not found at {path}. "
            "Run 'julia extract_optimal_actions.jl' to generate it. "
            "Skipping LP comparison."
        )
        return None
    df = pd.read_csv(path)
    n = int(df["prosumer"].max()) + 1
    T = int(df["hour"].max()) + 1
    actions = np.zeros((n, T), dtype=np.float32)
    for _, row in df.iterrows():
        actions[int(row["prosumer"]), int(row["hour"])] = float(row["action"])
    return actions


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _update_ppo(
    actor: PPOActor,
    critic: PPOCritic,
    buffer: RolloutBuffer,
    actor_optim: torch.optim.Optimizer,
    critic_optim: torch.optim.Optimizer,
    config: PPOConfig,
    device: torch.device,
) -> tuple[float, float]:
    """PPO update with clipped surrogate loss."""
    returns, advantages = buffer.compute_returns_and_advantages(
        config.gamma, config.gae_lambda, np.zeros(buffer.n_battery, dtype=np.float32)
    )
    buffer.returns = returns
    buffer.advantages = advantages

    actor_loss_accum = 0.0
    critic_loss_accum = 0.0

    for _ in range(config.n_updates):
        obs, actions, old_log_probs, returns, advantages = buffer.get_batch(config.batch_size, device)

        # Normalize advantages and returns
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Actor loss (PPO clipped surrogate)
        _, new_log_probs, _ = actor.sample(obs)
        ratio = torch.exp(new_log_probs - old_log_probs.unsqueeze(-1))
        clipped_ratio = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio)
        actor_loss = -torch.min(
            ratio * advantages.unsqueeze(-1),
            clipped_ratio * advantages.unsqueeze(-1),
        ).mean()

        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        # Critic loss
        values = critic(obs)
        critic_loss = F.mse_loss(values, returns)
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        actor_loss_accum += float(actor_loss.detach().cpu())
        critic_loss_accum += float(critic_loss.detach().cpu())

    return actor_loss_accum / config.n_updates, critic_loss_accum / config.n_updates


def _run_reference_episode(
    env: MARLEnvironment,
    actor: PPOActor,
    n: int,
    battery_idx: list[int],
    one_hots: np.ndarray,
    device: torch.device,
    *,
    lp_actions: np.ndarray | None = None,
) -> tuple[float, np.ndarray, float, np.ndarray, np.ndarray]:
    """Run one episode on the reference scenario using deterministic (or LP) actions."""
    obs_dict, _ = env.reset(options={"reference": True})
    obs = [obs_dict[f"consumer_{i}"] for i in range(n)]
    total_reward = 0.0
    total_discharge_bonus = 0.0
    hourly_import: list[float] = []
    p_ch_sum: list[float] = []
    p_dis_sum: list[float] = []
    n_battery = len(battery_idx)
    t = 0

    while env.agents:
        actions_dict: dict[str, np.ndarray] = {
            f"consumer_{i}": np.zeros(1, dtype=np.float32) for i in range(n) if env.E_max[i] == 0
        }
        if lp_actions is not None:
            for i in range(n):
                actions_dict[f"consumer_{i}"] = np.array([lp_actions[i, t]], dtype=np.float32)
        else:
            obs_batch = np.stack([
                np.concatenate([obs[battery_idx[j]], one_hots[j]]) for j in range(n_battery)
            ])
            obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
            with torch.no_grad():
                _, _, det_a = actor.sample(obs_t)
            all_actions = det_a.cpu().numpy()
            for j, i in enumerate(battery_idx):
                actions_dict[f"consumer_{i}"] = all_actions[j]

        next_obs_dict, rewards, _, _, infos = env.step(actions_dict)
        total_reward += sum(rewards.values())
        hourly_import.append(float(infos["consumer_0"]["total_import"]))
        p_ch_sum.append(float(sum(info["p_ch"] for info in infos.values())))
        p_dis_sum.append(float(sum(info["p_dis"] for info in infos.values())))
        viol = float(infos["consumer_0"]["capacity_violation"])
        if viol > 1e-4:
            total_discharge_bonus += env.violation_discharge_reward * sum(
                infos[f"consumer_{i}"]["p_dis"] for i in range(n)
            )
        obs = [next_obs_dict[f"consumer_{i}"] for i in range(n)]
        t += 1

    return (
        total_reward,
        np.asarray(hourly_import, dtype=np.float32),
        total_discharge_bonus,
        np.asarray(p_ch_sum, dtype=np.float32),
        np.asarray(p_dis_sum, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_ppo(config: PPOConfig) -> tuple[list[float], list[BenchmarkRollout], np.ndarray]:
    """Train PPO policy for decentralized energy-community MARL."""
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.use_wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed. Run: pip install wandb")
        run_name = (
            f"ppo_ep{config.episodes}_bs{config.batch_size}_"
            f"bonus{config.capacity_bonus}_alpha{config.alpha_grid}_"
            f"seed{config.seed}"
        )
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config=dataclasses.asdict(config),
        )

    env = MARLEnvironment(
        n_prosumers=config.n_prosumers,
        T=config.horizon,
        data_dir=config.data_dir,
        capacity_bonus=config.capacity_bonus,
        randomize_scenarios=config.randomize_scenarios,
        demand_noise_std=config.demand_noise_std,
        scenario_seed=config.seed,
        alpha_grid=config.alpha_grid,
        violation_discharge_reward=config.violation_discharge_reward,
    )
    n = env._n
    obs_dim = 6 * config.horizon + 2

    battery_idx: list[int] = [i for i in range(n) if env.E_max[i] > 0]
    n_battery = len(battery_idx)
    print(f"Battery agents: {battery_idx}  ({n_battery}/{n} prosumers)")

    one_hots = np.eye(n_battery, dtype=np.float32)

    action_dim = 1
    action_low = np.array([-1.0], dtype=np.float32)
    action_high = np.array([1.0], dtype=np.float32)
    net_obs_dim = obs_dim + n_battery

    actor = PPOActor(net_obs_dim, action_dim, action_low, action_high, config.hidden_dim).to(device)
    critic = PPOCritic(net_obs_dim, config.hidden_dim).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)

    # Load LP baseline (optional — requires running extract_optimal_actions.jl first)
    lp_actions = _load_lp_actions(config.data_dir)
    lp_baseline: float | None = None
    lp_reference_import: np.ndarray | None = None
    lp_p_ch: np.ndarray | None = None
    lp_p_dis: np.ndarray | None = None
    if lp_actions is not None:
        lp_baseline, lp_reference_import, _, lp_p_ch, lp_p_dis = _run_reference_episode(
            env, actor, n, battery_idx, one_hots, device, lp_actions=lp_actions,
        )
        print(f"LP baseline reward (reference day): {lp_baseline:.4f}")

    ref_cap: np.ndarray = env._ref_cap.copy()

    episode_returns: list[float] = []
    benchmark_history: list[BenchmarkRollout] = []

    t_reset = t_step = t_update = t_benchmark = 0.0

    for episode in range(config.episodes):
        _t = time.perf_counter()
        obs_dict, _ = env.reset()
        t_reset += time.perf_counter() - _t

        obs = [obs_dict[f"consumer_{i}"] for i in range(n)]
        buffer = RolloutBuffer(config.horizon, n_battery, net_obs_dim, action_dim)
        ep_return = 0.0

        step_idx = 0
        while env.agents:
            actions_dict: dict[str, np.ndarray] = {
                f"consumer_{i}": np.zeros(1, dtype=np.float32) for i in range(n) if env.E_max[i] == 0
            }

            # Collect trajectory step
            obs_batch = np.stack([
                np.concatenate([obs[battery_idx[j]], one_hots[j]]) for j in range(n_battery)
            ])
            obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
            with torch.no_grad():
                actions, log_probs, _ = actor.sample(obs_t)
                values = critic(obs_t)
            all_actions = actions.cpu().numpy()
            all_log_probs = log_probs.squeeze(-1).cpu().numpy()
            all_values = values.cpu().numpy()

            for j, i in enumerate(battery_idx):
                actions_dict[f"consumer_{i}"] = all_actions[j]

            _t = time.perf_counter()
            next_obs_dict, rewards, terms, _, _ = env.step(actions_dict)
            t_step += time.perf_counter() - _t

            done = any(terms.values())
            ep_return += sum(rewards.values())

            # Store trajectory
            rewards_batch = np.array([rewards[f"consumer_{battery_idx[j]}"] for j in range(n_battery)])
            dones_batch = np.full(n_battery, done, dtype=np.float32)

            next_obs = [next_obs_dict[f"consumer_{i}"] for i in range(n)]
            next_obs_batch = np.stack([
                np.concatenate([next_obs[battery_idx[j]], one_hots[j]]) for j in range(n_battery)
            ])

            buffer.add(
                obs_batch,
                all_actions,
                rewards_batch,
                all_values,
                all_log_probs,
                dones_batch,
            )

            obs = next_obs
            step_idx += 1

        # Compute final values for advantage estimation
        with torch.no_grad():
            final_obs_batch = np.stack([
                np.concatenate([obs[battery_idx[j]], one_hots[j]]) for j in range(n_battery)
            ])
            final_obs_t = torch.as_tensor(final_obs_batch, dtype=torch.float32, device=device)
            final_values = critic(final_obs_t).cpu().numpy()

        # PPO update
        _t = time.perf_counter()
        actor_loss, critic_loss = _update_ppo(
            actor, critic, buffer, actor_optim, critic_optim, config, device
        )
        t_update += time.perf_counter() - _t

        avg_return = ep_return / (n * config.horizon)
        episode_returns.append(avg_return)
        print(
            f"Episode {episode + 1:4d}/{config.episodes}: "
            f"avg_return={avg_return:10.4f}  actor_loss={actor_loss:8.4f}  critic_loss={critic_loss:8.4f}"
        )
        if config.use_wandb and wandb is not None:
            wandb.log({
                "episode": episode + 1,
                "avg_return": avg_return,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
            })

        if (episode + 1) % config.benchmark_interval == 0:
            _t = time.perf_counter()
            policy_reward, policy_import, policy_discharge_bonus, policy_p_ch, policy_p_dis = _run_reference_episode(
                env, actor, n, battery_idx, one_hots, device,
            )
            t_benchmark += time.perf_counter() - _t
            benchmark_history.append(
                BenchmarkRollout(
                    episode=episode + 1,
                    policy_reward=policy_reward,
                    policy_import=policy_import,
                    total_charging_reward=policy_discharge_bonus,
                    policy_p_ch=policy_p_ch,
                    policy_p_dis=policy_p_dis,
                    lp_reward=lp_baseline,
                    lp_import=lp_reference_import,
                    lp_p_ch=lp_p_ch,
                    lp_p_dis=lp_p_dis,
                )
            )
            total_import = float(policy_import.sum())
            lp_import_str = f"{float(lp_reference_import.sum()):.2f}" if lp_reference_import is not None else "N/A"
            lp_pct = f"  (LP return: {lp_baseline:.1f})" if lp_baseline else ""
            print(
                f"  [Benchmark ep {episode + 1}]"
                f"  reward={policy_reward:.2f}{lp_pct}"
                f"  import={total_import:.2f} kW (LP: {lp_import_str} kW)"
                f"  p_ch={float(policy_p_ch.sum()):.2f}  p_dis={float(policy_p_dis.sum()):.2f} kW"
            )
            if config.use_wandb and wandb is not None:
                bm_log: dict = {
                    "episode": episode + 1,
                    "benchmark/policy_reward": policy_reward,
                    "benchmark/total_import_kw": total_import,
                    "benchmark/p_ch_kw": float(policy_p_ch.sum()),
                    "benchmark/p_dis_kw": float(policy_p_dis.sum()),
                }
                if lp_baseline is not None:
                    bm_log["benchmark/lp_reward"] = lp_baseline
                    bm_log["benchmark/pct_of_lp"] = 100.0 * policy_reward / lp_baseline
                wandb.log(bm_log)

    total_t = t_reset + t_step + t_update + t_benchmark
    rows = [
        ("env.reset()",        t_reset),
        ("env.step()",         t_step),
        ("PPO update",         t_update),
        ("benchmark rollouts", t_benchmark),
    ]
    print("\n─── Timing breakdown ────────────────────────────")
    for label, secs in rows:
        pct = 100 * secs / total_t if total_t > 0 else 0.0
        print(f"  {label:<22s} {secs:7.2f}s  ({pct:5.1f}%)")
    print(f"  {'TOTAL':<22s} {total_t:7.2f}s")
    print("─────────────────────────────────────────────────")

    if config.use_wandb and wandb is not None:
        wandb.finish()

    return episode_returns, benchmark_history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_returns(episode_returns: list[float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, len(episode_returns) + 1)
    y = np.asarray(episode_returns, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, label="Avg per-agent per-step return", linewidth=1.5)
    if len(y) >= 10:
        win = min(20, len(y))
        ma = np.convolve(y, np.ones(win) / win, mode="valid")
        ax.plot(np.arange(win, len(y) + 1), ma, label=f"Moving average ({win})", linewidth=2.0)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average return (DKK per agent per step)")
    ax.set_title("PPO: Per-Agent Average Return")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_benchmark_actions(
    benchmark_history: list[BenchmarkRollout],
    out_path: Path,
    capacity_line: np.ndarray | None = None,
) -> None:
    if not benchmark_history:
        return

    n_panels = len(benchmark_history)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), squeeze=False)
    x = np.arange(1, len(benchmark_history[0].policy_import) + 1)

    for idx, record in enumerate(benchmark_history):
        ax = axes[0, idx]
        ax.bar(x, record.policy_import, width=0.9, color="#2E86AB", label=f"Policy ep {record.episode}")
        if record.policy_p_ch is not None:
            lbl_ch = "Sum battery charge" if idx == 0 else None
            ax.scatter(x, record.policy_p_ch, color="#2CA02C", marker='o', s=30, label=lbl_ch)
        if record.policy_p_dis is not None:
            lbl_dis = "Sum battery discharge" if idx == 0 else None
            ax.scatter(x, record.policy_p_dis, color="#D62728", marker='x', s=40, label=lbl_dis)
        if capacity_line is not None:
            ax.plot(x, capacity_line, color="black", linestyle="--", linewidth=1.5, label="Capacity limit")
        ax.set_title(f"Benchmark ep {record.episode}")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Community import [kW]")
        ax.set_xlim(1, len(x))
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)))

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO for decentralized energy-community MARL")
    parser.add_argument("--n-prosumers", type=int, default=14)
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--capacity-bonus", type=float, default=10.0)
    parser.add_argument("--data-dir", type=str, default="Data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-updates", type=int, default=5,
                        help="Number of PPO update steps per episode")
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.001)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--plot-path", type=str, default="Figures/ppo_returns.png")
    parser.add_argument("--benchmark-actions-path", type=str, default="Figures/ppo_benchmark_actions.png")
    parser.add_argument("--no-randomize-scenarios", action="store_true")
    parser.add_argument("--demand-noise-std", type=float, default=0.05)
    parser.add_argument("--benchmark-interval", type=int, default=500)
    parser.add_argument("--alpha-grid", type=float, default=75.0)
    parser.add_argument("--violation-discharge-reward", type=float, default=0.0)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="marl-energy")
    args = parser.parse_args()

    config = PPOConfig(
        n_prosumers=args.n_prosumers,
        episodes=args.episodes,
        horizon=args.horizon,
        capacity_bonus=args.capacity_bonus,
        data_dir=args.data_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        n_updates=args.n_updates,
        clip_ratio=args.clip_ratio,
        entropy_coeff=args.entropy_coeff,
        gae_lambda=args.gae_lambda,
        randomize_scenarios=not args.no_randomize_scenarios,
        demand_noise_std=args.demand_noise_std,
        benchmark_interval=args.benchmark_interval,
        alpha_grid=args.alpha_grid,
        violation_discharge_reward=args.violation_discharge_reward,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )

    episode_returns, benchmark_history = train_ppo(config)
    plot_path = Path(args.plot_path)
    plot_returns(episode_returns, plot_path)
    benchmark_actions_path = Path(args.benchmark_actions_path)
    plot_benchmark_actions(
        benchmark_history,
        benchmark_actions_path,
        capacity_line=benchmark_history[0].policy_import if benchmark_history else None,
    )
    print(f"Saved return plot to: {plot_path}")
    if benchmark_history:
        print(f"Saved benchmark actions plot to: {benchmark_actions_path}")


if __name__ == "__main__":
    main()
