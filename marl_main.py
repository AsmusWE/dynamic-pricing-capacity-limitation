from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from pathlib import Path
import time
from typing import cast

import hydra
from omegaconf import DictConfig
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
    total_charging_reward: float
    policy_p_ch: np.ndarray
    policy_p_dis: np.ndarray
    lp_reward: float | None
    lp_import: np.ndarray | None
    lp_p_ch: np.ndarray | None
    lp_p_dis: np.ndarray | None


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
    gamma: float = 0.999
    tau: float = 0.005
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    alpha_lr: float = 5e-4
    batch_size: int = 256
    buffer_size: int = 100_000
    warmup_steps: int = 500
    updates_per_step: int = 1
    hidden_dim: int = 256
    target_entropy_multiplier: float = 2.0
    seed: int = 42
    randomize_scenarios: bool = True
    demand_noise_std: float = 0.05
    benchmark_interval: int = 50
    alpha_grid: float = 75.0
    violation_discharge_reward: float = 0.0
    use_wandb: bool = False
    wandb_project: str = "marl-energy"
    wandb_run_name: str = ""  # empty → auto-generate from hyperparameters
    plot_path: str = "Figures/sac_returns.png"
    benchmark_actions_path: str = "Figures/sac_benchmark_actions.png"


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
    actor: Actor,
    n: int,
    battery_idx: list[int],
    one_hots: np.ndarray,
    device: torch.device,
    *,
    lp_actions: np.ndarray | None = None,
) -> tuple[float, np.ndarray, float, np.ndarray, np.ndarray, float]:
    """Run one episode on the reference scenario using deterministic actions.

    Returns: (total_reward, hourly_import, total_discharge_bonus, p_ch_per_hour, p_dis_per_hour, total_violation)
    """
    obs_dict, _ = env.reset(options={"reference": True})
    obs = [obs_dict[f"consumer_{i}"] for i in range(n)]
    total_reward = 0.0
    total_discharge_bonus = 0.0
    total_violation = 0.0
    hourly_import: list[float] = []
    p_ch_sum: list[float] = []
    p_dis_sum: list[float] = []
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
                np.concatenate([obs[battery_idx[j]], one_hots[j]]) for j in range(len(battery_idx))
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
        total_violation += viol
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
        total_violation,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_independent_sac(config: SACConfig) -> tuple[list[float], list[BenchmarkRollout]]:
    """Train a shared SAC policy for all battery agents (CTDE).

    Agents are distinguished by a one-hot identity vector appended to their observation.
    A single actor-critic pair and replay buffer are used, reducing gradient steps from
    n_battery per env-step to 1 — roughly an n_battery-fold speedup on the SAC update.
    """
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.use_wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed. Run: pip install wandb")
        auto_name = (
            f"ep{config.episodes}_bs{config.batch_size}_"
            f"bonus{config.capacity_bonus}_alpha{config.alpha_grid}_"
            f"vdr{config.violation_discharge_reward}_seed{config.seed}"
        )
        run_name = config.wandb_run_name or auto_name
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
    obs_dim = 6 * config.horizon + 2  # 146 for T=24

    # Only agents with a battery have meaningful actions; skip the rest entirely.
    battery_idx: list[int] = [i for i in range(n) if env.E_max[i] > 0]
    n_battery = len(battery_idx)
    print(f"Battery agents: {battery_idx}  ({n_battery}/{n} prosumers)")

    # One-hot identity per battery agent (row j → identity for battery_idx[j])
    one_hots = np.eye(n_battery, dtype=np.float32)

    action_dim = 1
    action_low = np.array([-1.0], dtype=np.float32)
    action_high = np.array([1.0], dtype=np.float32)
    # Network obs = env obs (146) + one-hot agent identity (n_battery)
    net_obs_dim = obs_dim + n_battery

    target_entropy = -float(action_dim) * config.target_entropy_multiplier

    # Shared actor-critic: all battery agents use the same weights
    actor = Actor(net_obs_dim, action_dim, action_low, action_high, config.hidden_dim).to(device)
    critic = Critic(net_obs_dim, action_dim, config.hidden_dim).to(device)
    target_critic = copy.deepcopy(critic)
    log_alpha = torch.tensor(0.0, device=device, requires_grad=True)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
    alpha_optim = torch.optim.Adam([log_alpha], lr=config.alpha_lr)

    # Single shared replay buffer (all agents' transitions mixed together)
    buffer = ReplayBuffer(config.buffer_size, net_obs_dim, action_dim)

    # Load LP baseline (optional — requires running extract_optimal_actions.jl first)
    lp_actions = _load_lp_actions(config.data_dir)
    lp_baseline: float | None = None
    lp_reference_import: np.ndarray | None = None
    lp_p_ch: np.ndarray | None = None
    lp_p_dis: np.ndarray | None = None
    if lp_actions is not None:
        lp_baseline, lp_reference_import, _, lp_p_ch, lp_p_dis, _ = _run_reference_episode(
            env, actor, n, battery_idx, one_hots, device, lp_actions=lp_actions,
        )
        print(f"LP baseline reward (reference day): {lp_baseline:.4f}")

    total_steps = 0
    episode_returns: list[float] = []
    benchmark_history: list[BenchmarkRollout] = []

    t_reset = t_obs = t_action = t_step = t_buffer = t_update = t_benchmark = 0.0

    for episode in range(config.episodes):
        _t = time.perf_counter()
        obs_dict, _ = env.reset()
        t_reset += time.perf_counter() - _t

        _t = time.perf_counter()
        obs = [obs_dict[f"consumer_{i}"] for i in range(n)]
        t_obs += time.perf_counter() - _t
        ep_return = 0.0
        ep_violation = 0.0

        while env.agents:
            # Battery-less agents always receive action=0
            actions_dict: dict[str, np.ndarray] = {
                f"consumer_{i}": np.zeros(1, dtype=np.float32) for i in range(n) if env.E_max[i] == 0
            }

            _t = time.perf_counter()
            if total_steps < config.warmup_steps:
                for i in battery_idx:
                    actions_dict[f"consumer_{i}"] = env.action_space(f"consumer_{i}").sample()
            else:
                obs_batch = np.stack([
                    np.concatenate([obs[battery_idx[j]], one_hots[j]]) for j in range(n_battery)
                ])
                obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
                with torch.no_grad():
                    a_batch, _, _ = actor.sample(obs_t)
                all_actions = a_batch.cpu().numpy()
                for j, i in enumerate(battery_idx):
                    actions_dict[f"consumer_{i}"] = all_actions[j]
            t_action += time.perf_counter() - _t

            _t = time.perf_counter()
            next_obs_dict, rewards, terms, _, step_infos = env.step(actions_dict)
            t_step += time.perf_counter() - _t

            done = any(terms.values())
            ep_return += sum(rewards.values())
            ep_violation += step_infos["consumer_0"]["capacity_violation"]

            _t = time.perf_counter()
            next_obs = [next_obs_dict[f"consumer_{i}"] for i in range(n)]
            for j, i in enumerate(battery_idx):
                aug_obs = np.concatenate([obs[i], one_hots[j]])
                aug_next = np.concatenate([next_obs[i], one_hots[j]])
                buffer.add(aug_obs, actions_dict[f"consumer_{i}"], rewards[f"consumer_{i}"], aug_next, done)
            t_buffer += time.perf_counter() - _t

            obs = next_obs
            total_steps += 1

            _t = time.perf_counter()
            if total_steps > config.warmup_steps and buffer.size >= config.batch_size:
                for _ in range(config.updates_per_step):
                    _update_agent(
                        actor, critic, target_critic,
                        buffer, log_alpha,
                        actor_optim, critic_optim, alpha_optim,
                        config, target_entropy, device,
                    )
            t_update += time.perf_counter() - _t

        avg_return = ep_return / (n * config.horizon)
        episode_returns.append(avg_return)
        alpha_val = float(log_alpha.exp().detach().cpu())
        print(
            f"Episode {episode + 1:4d}/{config.episodes}: "
            f"avg_return={avg_return:10.4f}  alpha={alpha_val:8.4f}  steps={total_steps}"
        )
        if config.use_wandb and wandb is not None:
            wandb.log({"avg_return": avg_return, "alpha": alpha_val, "capacity_violation": ep_violation})

        if (episode + 1) % config.benchmark_interval == 0:
            _t = time.perf_counter()
            policy_reward, policy_import, policy_discharge_bonus, policy_p_ch, policy_p_dis, policy_violation = _run_reference_episode(
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
                    lp_p_ch=(lp_p_ch if lp_actions is not None else None),
                    lp_p_dis=(lp_p_dis if lp_actions is not None else None),
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
                    "benchmark/policy_reward": policy_reward,
                    "benchmark/total_import_kw": total_import,
                    "benchmark/p_ch_kw": float(policy_p_ch.sum()),
                    "benchmark/p_dis_kw": float(policy_p_dis.sum()),
                    "benchmark/capacity_violation": policy_violation,
                }
                if lp_baseline is not None:
                    bm_log["benchmark/lp_reward"] = lp_baseline
                    bm_log["benchmark/pct_of_lp"] = 100.0 * policy_reward / lp_baseline
                    bm_log["benchmark/optimality_gap"] = lp_baseline - policy_reward
                wandb.log(bm_log)

    total_t = t_reset + t_obs + t_action + t_step + t_buffer + t_update + t_benchmark
    rows = [
        ("env.reset()",        t_reset),
        ("obs build",          t_obs),
        ("action selection",   t_action),
        ("env.step()",         t_step),
        ("buffer add",         t_buffer),
        ("SAC update",         t_update),
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
        # plot summed battery charge/discharge as dots (only label on first panel)
        if hasattr(record, 'policy_p_ch') and record.policy_p_ch is not None:
            lbl_ch = "Sum battery charge" if idx == 0 else None
            ax.scatter(x, record.policy_p_ch, color="#2CA02C", marker='o', s=30, label=lbl_ch)
        if hasattr(record, 'policy_p_dis') and record.policy_p_dis is not None:
            lbl_dis = "Sum battery discharge" if idx == 0 else None
            ax.scatter(x, record.policy_p_dis, color="#D62728", marker='x', s=40, label=lbl_dis)
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
        # LP battery charge/discharge dots if available
        lp_p_ch = benchmark_history[0].lp_p_ch
        lp_p_dis = benchmark_history[0].lp_p_dis
        if lp_p_ch is not None:
            ax_opt.scatter(x, lp_p_ch, color="#2CA02C", marker='o', s=30, label="LP sum battery charge")
        if lp_p_dis is not None:
            ax_opt.scatter(x, lp_p_dis, color="#D62728", marker='x', s=40, label="LP sum battery discharge")
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


@hydra.main(config_path="configs", config_name="config_prod", version_base=None)
def main(cfg: DictConfig) -> None:
    config = SACConfig(
        n_prosumers=cfg.n_prosumers,
        episodes=cfg.episodes,
        horizon=cfg.horizon,
        capacity_bonus=cfg.capacity_bonus,
        data_dir=cfg.data_dir,
        gamma=cfg.gamma,
        tau=cfg.tau,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        alpha_lr=cfg.alpha_lr,
        batch_size=cfg.batch_size,
        buffer_size=cfg.buffer_size,
        warmup_steps=cfg.warmup_steps,
        updates_per_step=cfg.updates_per_step,
        hidden_dim=cfg.hidden_dim,
        target_entropy_multiplier=cfg.target_entropy_multiplier,
        seed=cfg.seed,
        randomize_scenarios=cfg.randomize_scenarios,
        demand_noise_std=cfg.demand_noise_std,
        benchmark_interval=cfg.benchmark_interval,
        alpha_grid=cfg.alpha_grid,
        violation_discharge_reward=cfg.violation_discharge_reward,
        use_wandb=cfg.use_wandb,
        wandb_project=cfg.wandb_project,
        wandb_run_name=cfg.wandb_run_name,
        plot_path=cfg.plot_path,
        benchmark_actions_path=cfg.benchmark_actions_path,
    )

    episode_returns, benchmark_history = train_independent_sac(config)
    lp_baseline = benchmark_history[0].lp_reward if benchmark_history else None
    plot_path = Path(cfg.plot_path)
    plot_returns(episode_returns, plot_path, benchmark_history=benchmark_history, lp_baseline=lp_baseline)
    benchmark_actions_path = Path(cfg.benchmark_actions_path)
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
