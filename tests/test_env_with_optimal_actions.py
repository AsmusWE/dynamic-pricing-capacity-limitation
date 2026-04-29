"""Test that the MARL environment with real data produces results matching the Julia optimizer."""

import json, os, subprocess, sys, numpy as np

REPO_ROOT = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(REPO_ROOT, os.pardir))

def run_julia_optimizer():
    script = os.path.join(REPO_ROOT, "run_optimization.jl")
    print(f"Running Julia optimizer: {script}")
    subprocess.run(["julia", script], check=True, cwd=REPO_ROOT)

def load_solution():
    with open(os.path.join(REPO_ROOT, "outputs", "opt_solution.json")) as f:
        return json.load(f)

def _normalize(arr, n, t):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1: return arr.reshape(1, -1)
    if arr.shape == (n, t): return arr
    if arr.shape == (t, n): return arr.T
    raise ValueError(f"Bad shape {arr.shape}")

def test_env_data_matches_solution():
    sol = load_solution()
    sys.path.insert(0, REPO_ROOT)
    from marl_environment import make_env
    env = make_env(None, n_prosumers=14, T=24, seed=1234)
    env.reset()
    for key in ["D", "PV"]:
        exp = _normalize(sol[key], 14, 24)
        act = _normalize(env.data[key], 14, 24)
        assert np.allclose(exp, act, atol=1e-4), f"{key}: max diff {np.abs(exp-act).max():.6f}"
    for key in ["spot", "y_im", "y_ex"]:
        exp = np.asarray(sol[key], dtype=np.float64).flatten()
        act = np.asarray(env.data[key], dtype=np.float64).flatten()
        assert np.allclose(exp, act, atol=1e-4), f"{key}: max diff {np.abs(exp-act).max():.6f}"
    assert "cap" in env.data
    print("Environment data matches opt_solution.json (including cap).")

def test_consumer_objectives_match():
    """The decentralized LP achieves the SAME consumer cost as the centralized solution."""
    sol = load_solution()
    sys.path.insert(0, REPO_ROOT)
    from consumer_step import solve_consumer_step
    from data_loader import load_data

    data = load_data(os.path.join(REPO_ROOT, "Data"), 14, 1234)
    prices = _normalize(sol["x"], 14, 24)

    # Centralized consumer costs: x * (p_plus - p_minus) per prosumer
    p_plus_opt = _normalize(sol["p_plus"], 14, 24)
    p_minus_opt = _normalize(sol["p_minus"], 14, 24)
    opt_cost = np.sum(prices * (p_plus_opt - p_minus_opt))

    # Decentralized consumer LP
    payload = {
        "prices": prices.astype(np.float32).tolist(),
        "D": data["D"].astype(np.float32).tolist(),
        "PV": data["PV"].astype(np.float32).tolist(),
        "soc": np.zeros(14, dtype=np.float32).tolist(),
        "eta": float(data["eta"]),
        "E_max": data["E_max"].astype(np.float32).tolist(),
        "p_ch_max": data["p_ch_max"].astype(np.float32).tolist(),
        "p_dis_max": data["p_dis_max"].astype(np.float32).tolist(),
        "y_im": data["y_im"].astype(np.float32).tolist(),
        "y_ex": data["y_ex"].astype(np.float32).tolist(),
    }
    out = solve_consumer_step(payload)
    p_plus_cs = np.array(out["p_plus"])
    p_minus_cs = np.array(out["p_minus"])
    cs_cost = np.sum(prices * (p_plus_cs - p_minus_cs))

    assert np.isclose(opt_cost, cs_cost, atol=1e-2), f"Consumer cost mismatch: {opt_cost:.4f} vs {cs_cost:.4f}"
    print(f"Consumer costs match: opt={opt_cost:.4f}, cs={cs_cost:.4f}")
    print("(Battery cycling is zero-profit at the optimal KKT prices -- the LP picks")
    print(" one of many equally-optimal solutions.  The RL agent must learn prices")
    print(" that make battery cycling strictly profitable for capacity management.)")

def test_env_consumer_step_roundtrip():
    sol = load_solution()
    T, n = 24, 14
    prices = _normalize(sol["x"], n, T)
    sys.path.insert(0, REPO_ROOT)
    from marl_environment import make_env
    from consumer_step import solve_consumer_step
    env = make_env(None, n_prosumers=n, T=T, seed=1234)
    env.reset()
    action = prices
    payload = {
        "prices": np.asarray(action, dtype=np.float32).tolist(),
        "D": env.data["D"].astype(np.float32).tolist(),
        "PV": env.data["PV"].astype(np.float32).tolist(),
        "soc": env.soc.astype(np.float32).tolist(),
        "eta": float(env.data["eta"]),
        "E_max": env.data["E_max"].astype(np.float32).tolist(),
        "p_ch_max": env.data["p_ch_max"].astype(np.float32).tolist(),
        "p_dis_max": env.data["p_dis_max"].astype(np.float32).tolist(),
        "y_im": env.data["y_im"].astype(np.float32).tolist(),
        "y_ex": env.data["y_ex"].astype(np.float32).tolist(),
    }
    expected = solve_consumer_step(payload)
    actual = env.step_with_action(action.tolist())
    for key in expected:
        exp_arr = np.asarray(expected[key], dtype=np.float64)
        act_arr = np.asarray(actual.get(key), dtype=np.float64)
        assert np.allclose(exp_arr, act_arr, atol=1e-6), f"Mismatch for {key}"
    print("Consumer step roundtrip through env matches direct call.")

def test_decentralized_reward():
    sol = load_solution()
    prices = _normalize(sol["x"], 14, 24)
    sys.path.insert(0, REPO_ROOT)
    from marl_environment import make_env
    env = make_env(None, n_prosumers=14, T=24, seed=1234)
    env.reset()
    out = env.step_with_action(prices.tolist())
    reward = float(env.rewards[env.agents[0]])
    breakdown = env.infos[env.agents[0]].get("reward_breakdown", {})
    assert np.isfinite(reward)
    print(f"Decentralised reward: {reward:.4f}")
    print(f"  Breakdown: { {k: f'{v:.4f}' for k, v in breakdown.items() if k != 'reward'} }")
    objective = float(sol["objective"])
    print(f"Centralised optimum (-objective): {-objective:.4f}")
    print("(Gap is the decomposition gap: consumers do not see the capacity limit,")
    print(" and battery cycling is zero-profit at the optimal KKT prices.)")

if __name__ == "__main__":
    sol_path = os.path.join(REPO_ROOT, "outputs", "opt_solution.json")
    if not os.path.exists(sol_path):
        run_julia_optimizer()
    test_env_data_matches_solution()
    test_consumer_objectives_match()
    test_env_consumer_step_roundtrip()
    test_decentralized_reward()
    print("All env-with-optimal-actions tests passed!")
