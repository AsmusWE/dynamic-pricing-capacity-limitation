from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from marl_environment import MARLEnvironment


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


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


@dataclass
class BenchmarkRollout:
    episode: int
    policy_reward: float
    policy_import: np.ndarray
    lp_reward: float | None
    lp_import: np.ndarray | None


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

        # Tanh correction for the reparameterized log-prob.
        log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - tanh_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        deterministic = torch.tanh(mu) * action_scale + action_bias
        return action, log_prob, deterministic


# ---------------------------------------------------------------------------
# SAC configuration
# ---------------------------------------------------------------------------


@dataclass
class SACConfig:
    n_prosumers: int = 14
    episodes: int = 200
    horizon: int = 24
    capacity_bonus: float = 5.0
    data_dir: str = "Data"
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    alpha_lr: float = 1e-3
    batch_size: int = 256
    buffer_size: int = 100_000
    warmup_steps: int = 500
    updates_per_step: int = 1
    hidden_dim: int = 256
    seed: int = 42
    randomize_scenarios: bool = True
    demand_noise_std: float = 0.05
    benchmark_interval: int = 50


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.mul_(1.0 - tau).add_(tau * s.data)


def _update_agent(
    actor: Actor,
    critic: Critic,
    target_critic: Critic,
    buffer: ReplayBuffer,
    log_alpha: torch.Tensor,
    actor_optim: torch.optim.Optimizer,
    critic_optim: torch.optim.Optimizer,
    alpha_optim: torch.optim.Optimizer,
    config: SACConfig,
    target_entropy: float,
    device: torch.device,
) -> None:
    o, a, r, no, d = buffer.sample(config.batch_size, device)

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


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _load_lp_actions(data_dir: str) -> np.ndarray | None:
    """Load pre-computed LP optimal actions from Data/optimal_actions.csv.

    Returns (n_prosumers, T) float32 array indexed [prosumer, hour], or None
    if the file does not exist.
    """
    path = Path(data_dir) / "optimal_actions.csv"
    if not path.exists():
        print(
            f"[Benchmark] optimal_actions.csv not found at {path}. "
            "Run 'julia extract_optimal_actions.jl' to generate it. "
            "Skipping LP comparison."
        )
        return None
    import pandas as pd
    df = pd.read_csv(path)
    n = int(df["prosumer"].max()) + 1
    T = int(df["hour"].max()) + 1
    actions = np.zeros((n, T), dtype=np.float32)
    for _, row in df.iterrows():
        actions[int(row["prosumer"]), int(row["hour"])] = float(row["action"])
    return actions


def _run_reference_episode(
    env: MARLEnvironment,
    actors: list,
    n: int,
    device: torch.device,
    *,
    deterministic: bool,
    lp_actions: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Run one episode on the reference scenario and return total reward and hourly import.

    If deterministic=True, uses the actor mean (no exploration).
    If lp_actions is provided, uses those instead of the policy.
    """
    obs_dict, _ = env.reset(options={"reference": True})
    obs = [obs_dict[f"consumer_{i}"] for i in range(n)]
    total_reward = 0.0
    hourly_import: list[float] = []
    t = 0
    while env.agents:
        actions_dict: dict[str, np.ndarray] = {}
        for i in range(n):
            if lp_actions is not None:
                a = np.array([lp_actions[i, t]], dtype=np.float32)
            else:
                obs_t = torch.as_tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    _, _, det_a = actors[i].sample(obs_t)
                if deterministic:
                    a = det_a.squeeze(0).cpu().numpy().astype(np.float32)
                else:
                    a = det_a.squeeze(0).cpu().numpy().astype(np.float32)
            actions_dict[f"consumer_{i}"] = a
        next_obs_dict, rewards, _, _, infos = env.step(actions_dict)
        total_reward += sum(rewards.values())
        hourly_import.append(float(sum(info["p_plus"] for info in infos.values())))
        obs = [next_obs_dict[f"consumer_{i}"] for i in range(n)]
        t += 1
    return total_reward, np.asarray(hourly_import, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_independent_sac(config: SACConfig) -> tuple[list[float], list[BenchmarkRollout]]:
    """Train N independent SAC agents, one per prosumer."""
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MARLEnvironment(
        n_prosumers=config.n_prosumers,
        T=config.horizon,
        data_dir=config.data_dir,
        capacity_bonus=config.capacity_bonus,
        randomize_scenarios=config.randomize_scenarios,
        demand_noise_std=config.demand_noise_std,
        scenario_seed=config.seed,
    )
    n = env._n
    obs_dim = 6 * config.horizon + 2  # 146 for T=24
    action_dim = 1
    action_low = np.array([-1.0], dtype=np.float32)
    action_high = np.array([1.0], dtype=np.float32)
    target_entropy = -float(action_dim)*10

    actors = [
        Actor(obs_dim, action_dim, action_low, action_high, config.hidden_dim).to(device)
        for _ in range(n)
    ]
    critics = [Critic(obs_dim, action_dim, config.hidden_dim).to(device) for _ in range(n)]
    target_critics = [copy.deepcopy(critics[i]) for i in range(n)]

    log_alphas = [torch.tensor(0.0, device=device, requires_grad=True) for _ in range(n)]
    actor_optims = [torch.optim.Adam(actors[i].parameters(), lr=config.actor_lr) for i in range(n)]
    critic_optims = [torch.optim.Adam(critics[i].parameters(), lr=config.critic_lr) for i in range(n)]
    alpha_optims = [torch.optim.Adam([log_alphas[i]], lr=config.alpha_lr) for i in range(n)]

    buffers = [ReplayBuffer(config.buffer_size, obs_dim, action_dim) for _ in range(n)]

    # Load LP baseline (optional — requires running extract_optimal_actions.jl first)
    lp_actions = _load_lp_actions(config.data_dir)
    lp_baseline: float | None = None
    lp_reference_import: np.ndarray | None = None
    if lp_actions is not None:
        lp_baseline, lp_reference_import = _run_reference_episode(
            env, actors, n, device, deterministic=True, lp_actions=lp_actions
        )
        print(f"LP baseline reward (reference day): {lp_baseline:.4f}")

    total_steps = 0
    episode_returns: list[float] = []
    benchmark_history: list[BenchmarkRollout] = []

    for episode in range(config.episodes):
        obs_dict, _ = env.reset()
        obs = [obs_dict[f"consumer_{i}"] for i in range(n)]
        ep_return = 0.0

        while env.agents:
            # Collect actions for all agents
            actions_dict: dict[str, np.ndarray] = {}
            for i in range(n):
                if total_steps < config.warmup_steps:
                    a = env.action_space(f"consumer_{i}").sample()
                else:
                    obs_t = torch.as_tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        a_t, _, _ = actors[i].sample(obs_t)
                    a = a_t.squeeze(0).cpu().numpy().astype(np.float32)
                actions_dict[f"consumer_{i}"] = a

            next_obs_dict, rewards, terms, truncs, infos = env.step(actions_dict)
            done = any(terms.values())
            next_obs = [next_obs_dict[f"consumer_{i}"] for i in range(n)]

            for i in range(n):
                buffers[i].add(
                    obs[i],
                    actions_dict[f"consumer_{i}"],
                    rewards[f"consumer_{i}"],
                    next_obs[i],
                    done,
                )
                ep_return += rewards[f"consumer_{i}"]

            obs = next_obs
            total_steps += 1

            # Independent SAC update per agent
            if total_steps > config.warmup_steps and buffers[0].size >= config.batch_size:
                for _ in range(config.updates_per_step):
                    for i in range(n):
                        _update_agent(
                            actors[i], critics[i], target_critics[i],
                            buffers[i], log_alphas[i],
                            actor_optims[i], critic_optims[i], alpha_optims[i],
                            config, target_entropy, device,
                        )

        avg_return = ep_return / (n * config.horizon)
        episode_returns.append(avg_return)
        alpha_value = float(torch.stack([log_alpha.exp().detach() for log_alpha in log_alphas]).mean().cpu())
        print(
            f"Episode {episode + 1:4d}/{config.episodes}: "
            f"avg_return={avg_return:10.4f}  alpha={alpha_value:8.4f}  steps={total_steps}"
        )

        if (episode + 1) % config.benchmark_interval == 0:
            policy_reward, policy_import = _run_reference_episode(
                env, actors, n, device, deterministic=True
            )
            benchmark_history.append(
                BenchmarkRollout(
                    episode=episode + 1,
                    policy_reward=policy_reward,
                    policy_import=policy_import,
                    lp_reward=lp_baseline,
                    lp_import=lp_reference_import,
                )
            )
            if lp_baseline is not None:
                print(
                    f"  [Benchmark ep {episode + 1}] "
                    f"policy={policy_reward:.4f}  lp_baseline={lp_baseline:.4f}"
                )
            else:
                print(
                    f"  [Benchmark ep {episode + 1}] "
                    f"policy={policy_reward:.4f}  lp_baseline=N/A"
                )

    return episode_returns, benchmark_history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_returns(
    episode_returns: list[float],
    out_path: Path,
    benchmark_history: list[BenchmarkRollout] | None = None,
    lp_baseline: float | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, len(episode_returns) + 1)
    y = np.asarray(episode_returns, dtype=np.float64)

    has_benchmark = bool(benchmark_history)
    n_panels = 2 if has_benchmark else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels), squeeze=False)
    ax_train = axes[0, 0]

    ax_train.plot(x, y, label="Avg per-agent per-step return", linewidth=1.5)
    if len(y) >= 10:
        win = min(20, len(y))
        ma = np.convolve(y, np.ones(win) / win, mode="valid")
        ax_train.plot(np.arange(win, len(y) + 1), ma, label=f"Moving average ({win})", linewidth=2.0)
    ax_train.set_xlabel("Episode")
    ax_train.set_ylabel("Average return (DKK per agent per step)")
    ax_train.set_title("Independent SAC: Per-Agent Average Return")
    ax_train.grid(alpha=0.25)
    ax_train.legend()

    if has_benchmark:
        ax_bm = axes[1, 0]
        bm_episodes = [h.episode for h in benchmark_history]
        bm_rewards = [h.policy_reward for h in benchmark_history]

        if lp_baseline is not None:
            bm_values = bm_rewards
            ax_bm.axhline(lp_baseline, color="green", linestyle="--", linewidth=1.5, label="LP baseline reward")
            ax_bm.set_ylabel("Reward on reference day")
        else:
            bm_values = bm_rewards
            ax_bm.set_ylabel("Policy reward on reference day")

        ax_bm.plot(bm_episodes, bm_values, marker="o", linewidth=1.5, label="Policy benchmark")
        ax_bm.set_xlabel("Episode")
        ax_bm.set_title("Benchmark: Reference Day Performance")
        ax_bm.grid(alpha=0.25)
        ax_bm.legend()

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

    n_panels = len(benchmark_history) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), squeeze=False)
    x = np.arange(1, len(benchmark_history[0].policy_import) + 1)

    for idx, record in enumerate(benchmark_history):
        ax = axes[0, idx]
        ax.bar(x, record.policy_import, width=0.9, color="#2E86AB", label=f"Policy ep {record.episode}")
        if record.lp_import is not None:
            ax.plot(x, record.lp_import, color="#D1495B", linewidth=2.0, label="LP optimal")
        if capacity_line is not None:
            ax.plot(x, capacity_line, color="black", linestyle="--", linewidth=1.5, label="Capacity limit")
        ax.set_title(f"Benchmark ep {record.episode}")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Community import [kW]")
        ax.set_xlim(1, len(x))
        ax.grid(alpha=0.25)

    ax_opt = axes[0, -1]
    lp_import = benchmark_history[0].lp_import
    if lp_import is not None:
        ax_opt.bar(x, lp_import, width=0.9, color="#D1495B", label="LP optimal actions")
    if capacity_line is not None:
        ax_opt.plot(x, capacity_line, color="black", linestyle="--", linewidth=1.5, label="Capacity limit")
    ax_opt.set_title("LP optimal actions")
    ax_opt.set_xlabel("Hour")
    ax_opt.set_ylabel("Community import [kW]")
    ax_opt.set_xlim(1, len(x))
    ax_opt.grid(alpha=0.25)

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
    parser = argparse.ArgumentParser(description="Independent SAC for decentralized energy-community MARL")
    parser.add_argument("--n-prosumers", type=int, default=14)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--capacity-bonus", type=float, default=100.0)
    parser.add_argument("--data-dir", type=str, default="Data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--plot-path", type=str, default="Figures/independent_sac_returns.png")
    parser.add_argument("--benchmark-actions-path", type=str, default="Figures/independent_sac_benchmark_actions.png")
    parser.add_argument("--no-randomize-scenarios", action="store_true",
                        help="Disable scenario randomization (use fixed Aug-2 data every episode)")
    parser.add_argument("--demand-noise-std", type=float, default=0.05)
    parser.add_argument("--benchmark-interval", type=int, default=25)
    args = parser.parse_args()

    config = SACConfig(
        n_prosumers=args.n_prosumers,
        episodes=args.episodes,
        horizon=args.horizon,
        capacity_bonus=args.capacity_bonus,
        data_dir=args.data_dir,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        updates_per_step=args.updates_per_step,
        randomize_scenarios=not args.no_randomize_scenarios,
        demand_noise_std=args.demand_noise_std,
        benchmark_interval=args.benchmark_interval,
    )

    episode_returns, benchmark_history = train_independent_sac(config)
    lp_baseline = benchmark_history[0].lp_reward if benchmark_history else None
    plot_path = Path(args.plot_path)
    plot_returns(episode_returns, plot_path, benchmark_history=benchmark_history, lp_baseline=lp_baseline)
    benchmark_actions_path = Path(args.benchmark_actions_path)
    plot_benchmark_actions(
        benchmark_history,
        benchmark_actions_path,
        capacity_line=benchmark_history[0].lp_import if benchmark_history and benchmark_history[0].lp_import is not None else None,
    )
    print(f"Saved return plot to: {plot_path}")
    if benchmark_history:
        print(f"Saved benchmark actions plot to: {benchmark_actions_path}")


if __name__ == "__main__":
    main()
