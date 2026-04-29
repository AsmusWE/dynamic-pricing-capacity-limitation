"""Tests for the decentralized consumer-agent MARLEnvironment (ParallelEnv)."""
from __future__ import annotations

import numpy as np
import pytest

from marl_environment import MARLEnvironment

N = 14
T = 24
OBS_DIM = 6 * T + 2  # 146


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    return MARLEnvironment(n_prosumers=N, T=T, data_dir="Data", seed=1234)


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------


def _actions(value: float, n: int = N) -> dict[str, np.ndarray]:
    return {f"consumer_{i}": np.array([value], dtype=np.float32) for i in range(n)}


def zero_actions(n: int = N) -> dict[str, np.ndarray]:
    return _actions(0.0, n)


def charge_actions(n: int = N) -> dict[str, np.ndarray]:
    return _actions(1.0, n)


def discharge_actions(n: int = N) -> dict[str, np.ndarray]:
    return _actions(-1.0, n)


# ---------------------------------------------------------------------------
# Observation shape and space
# ---------------------------------------------------------------------------


def test_reset_obs_shape(env):
    obs, _ = env.reset(seed=0)
    for agent in env.agents:
        assert obs[agent].shape == (OBS_DIM,), f"Bad shape for {agent}: {obs[agent].shape}"


def test_obs_in_space_after_reset(env):
    obs, _ = env.reset(seed=0)
    for agent in env.agents:
        space = env.observation_space(agent)
        assert space.contains(obs[agent]), f"Obs out of declared space for {agent}"


def test_obs_in_space_mid_episode(env):
    env.reset(seed=0)
    for _ in range(T // 2):
        obs, _, _, _, _ = env.step(zero_actions())
    for agent in env.possible_agents:
        space = env.observation_space(agent)
        assert space.contains(obs[agent]), f"Mid-episode obs out of space for {agent}"


# ---------------------------------------------------------------------------
# Episode length
# ---------------------------------------------------------------------------


def test_episode_runs_exactly_T_steps(env):
    env.reset(seed=0)
    steps = 0
    while env.agents:
        env.step(zero_actions())
        steps += 1
    assert steps == T, f"Expected {T} steps, got {steps}"


def test_agents_empty_after_episode(env):
    env.reset(seed=0)
    while env.agents:
        env.step(zero_actions())
    assert env.agents == []


def test_terminations_true_on_last_step(env):
    env.reset(seed=0)
    terms = {}
    while env.agents:
        _, _, terms, _, _ = env.step(zero_actions())
    assert all(terms.values()), "All agents should be terminated after last step"


# ---------------------------------------------------------------------------
# SoC update
# ---------------------------------------------------------------------------


def _first_battery_agent(env) -> int:
    return next(i for i in range(N) if env.E_max[i] > 0)


def test_soc_increases_on_charge(env):
    env.reset(seed=0)
    i = _first_battery_agent(env)
    env.soc[i] = 0.0

    actions = zero_actions()
    actions[f"consumer_{i}"] = np.array([1.0], dtype=np.float32)
    env.step(actions)

    assert env.soc[i] > 0.0, "SoC should increase after charging"


def test_soc_decreases_on_discharge(env):
    env.reset(seed=0)
    i = _first_battery_agent(env)
    env.soc[i] = env.E_max[i]

    actions = zero_actions()
    actions[f"consumer_{i}"] = np.array([-1.0], dtype=np.float32)
    env.step(actions)

    assert env.soc[i] < env.E_max[i], "SoC should decrease after discharging"


def test_soc_does_not_exceed_E_max(env):
    env.reset(seed=0)
    i = _first_battery_agent(env)
    env.soc[i] = env.E_max[i]  # already full

    actions = zero_actions()
    actions[f"consumer_{i}"] = np.array([1.0], dtype=np.float32)
    env.step(actions)

    assert env.soc[i] <= env.E_max[i] + 1e-6, "SoC must not exceed E_max"


def test_soc_does_not_go_below_zero(env):
    env.reset(seed=0)
    i = _first_battery_agent(env)
    env.soc[i] = 0.0  # already empty

    actions = zero_actions()
    actions[f"consumer_{i}"] = np.array([-1.0], dtype=np.float32)
    env.step(actions)

    assert env.soc[i] >= -1e-6, "SoC must not go below 0"


def test_soc_update_matches_formula(env):
    """SoC update = soc + p_ch - p_dis (DC-side, no η in SoC equation)."""
    env.reset(seed=0)
    i = _first_battery_agent(env)
    env.soc[i] = env.E_max[i] * 0.5

    actions = zero_actions()
    actions[f"consumer_{i}"] = np.array([1.0], dtype=np.float32)

    soc_before = float(env.soc[i])
    _, _, _, _, infos = env.step(actions)

    p_ch = infos[f"consumer_{i}"]["p_ch"]
    p_dis = infos[f"consumer_{i}"]["p_dis"]
    expected_soc = np.clip(soc_before + p_ch - p_dis, 0.0, env.E_max[i])
    assert abs(env.soc[i] - expected_soc) < 1e-5, "SoC update formula mismatch"


# ---------------------------------------------------------------------------
# Battery-less agents
# ---------------------------------------------------------------------------


def test_zero_battery_agents_have_no_power_flow(env):
    """Agents with E_max=0 must produce p_ch=0 and p_dis=0 regardless of action."""
    zero_battery = [i for i in range(N) if env.E_max[i] == 0]
    assert len(zero_battery) > 0, "This test requires at least one battery-less prosumer"

    env.reset(seed=0)
    actions = {f"consumer_{i}": np.array([1.0], dtype=np.float32) for i in range(N)}
    _, _, _, _, infos = env.step(actions)

    for i in zero_battery:
        assert infos[f"consumer_{i}"]["p_ch"] == 0.0, f"Consumer {i} has no battery but p_ch > 0"
        assert infos[f"consumer_{i}"]["p_dis"] == 0.0, f"Consumer {i} has no battery but p_dis > 0"


def test_zero_battery_soc_stays_zero(env):
    zero_battery = [i for i in range(N) if env.E_max[i] == 0]
    env.reset(seed=0)
    env.step(charge_actions())
    for i in zero_battery:
        assert env.soc[i] == 0.0


# ---------------------------------------------------------------------------
# Reward structure
# ---------------------------------------------------------------------------


def test_reward_formula_no_violation(env):
    """reward[i] = -individual_cost[i] + capacity_bonus when no violation."""
    env.reset(seed=0)
    env.soc = env.E_max.copy()  # full batteries so discharge reduces import

    _, rewards, _, _, infos = env.step(discharge_actions())

    info0 = infos["consumer_0"]
    if info0["capacity_violation"] == 0.0:
        for i in range(N):
            expected = -infos[f"consumer_{i}"]["individual_cost"] + env.capacity_bonus
            assert abs(rewards[f"consumer_{i}"] - expected) < 1e-5, (
                f"Reward formula wrong for consumer_{i}: got {rewards[f'consumer_{i}']:.6f}, "
                f"expected {expected:.6f}"
            )


def test_reward_formula_with_violation(env):
    """reward[i] = -individual_cost[i] - alpha_grid * violation / N when cap exceeded."""
    env.reset(seed=0)
    env.soc = np.zeros(N, dtype=np.float32)  # empty → charging draws maximum from grid

    _, rewards, _, _, infos = env.step(charge_actions())

    info0 = infos["consumer_0"]
    violation = info0["capacity_violation"]
    if violation > 0.0:
        expected_collective = -env.alpha_grid * violation / N
        for i in range(N):
            expected = -infos[f"consumer_{i}"]["individual_cost"] + expected_collective
            assert abs(rewards[f"consumer_{i}"] - expected) < 1e-4, (
                f"Penalty reward formula wrong for consumer_{i}"
            )


def test_individual_cost_positive_when_importing(env):
    env.reset(seed=0)
    env.soc = np.zeros(N, dtype=np.float32)  # no stored energy, likely importing

    _, _, _, _, infos = env.step(zero_actions())

    for i in range(N):
        p_plus = infos[f"consumer_{i}"]["p_plus"]
        cost = infos[f"consumer_{i}"]["individual_cost"]
        if p_plus > 1e-9:
            assert cost > 0.0, f"Consumer {i} importing but individual_cost <= 0"


def test_individual_cost_negative_when_net_exporting(env):
    env.reset(seed=0)
    env.soc = env.E_max.copy()

    _, _, _, _, infos = env.step(discharge_actions())

    for i in range(N):
        p_plus = infos[f"consumer_{i}"]["p_plus"]
        p_minus = infos[f"consumer_{i}"]["p_minus"]
        cost = infos[f"consumer_{i}"]["individual_cost"]
        if p_minus > p_plus + 1e-9:
            assert cost < 0.0, f"Consumer {i} net-exporting but individual_cost >= 0"


def test_total_import_reported_correctly(env):
    env.reset(seed=0)
    _, _, _, _, infos = env.step(zero_actions())

    # All agents see the same total_import
    total_imports = {infos[f"consumer_{i}"]["total_import"] for i in range(N)}
    assert len(total_imports) == 1, "All agents should observe the same total_import"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_reset(env):
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    for agent in env.possible_agents:
        np.testing.assert_array_equal(obs1[agent], obs2[agent])


def test_different_seeds_give_different_initial_soc(env):
    if env.E_max.sum() == 0:
        pytest.skip("No batteries in this configuration")
    env.reset(seed=0)
    soc0 = env.soc.copy()
    env.reset(seed=999)
    soc1 = env.soc.copy()
    assert not np.allclose(soc0, soc1), "Different seeds should produce different initial SoC"


# ---------------------------------------------------------------------------
# PettingZoo API compliance
# ---------------------------------------------------------------------------


def test_parallel_env_api():
    """Verify the environment passes PettingZoo's official parallel API test."""
    from pettingzoo.test import parallel_api_test
    env = MARLEnvironment(n_prosumers=N, T=T, data_dir="Data", seed=1234)
    parallel_api_test(env, num_cycles=2)
