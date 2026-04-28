import subprocess
import json
import os
import numpy as np
import sys

REPO_ROOT = os.path.dirname(__file__)  # tests dir
REPO_ROOT = os.path.abspath(os.path.join(REPO_ROOT, os.pardir))

def run_julia_optimizer():
    script = os.path.join(REPO_ROOT, "run_optimization.jl")
    print(f"Running Julia optimizer: {script}")
    subprocess.run(["julia", script], check=True, cwd=REPO_ROOT)

def load_solution():
    path = os.path.join(REPO_ROOT, "outputs", "opt_solution.json")
    with open(path, "r") as f:
        sol = json.load(f)
    return sol

def normalize_price_matrix(x, T=24, n=14):
    arr = np.array(x)
    # Accept either (n, T) or (T, n) and normalize to (n, T).
    if arr.ndim != 2:
        raise ValueError(f"Unexpected price matrix rank: {arr.ndim}")
    if arr.shape[1] == T:
        return arr
    if arr.shape[0] == T:
        return arr.T
    raise ValueError(f"Unexpected price matrix shape: {arr.shape}")


def test_env_matches_optimizer():
    run_julia_optimizer()
    sol = load_solution()
    # read prices
    T = 24
    n = 14
    prices = normalize_price_matrix(sol["x"], T=T, n=n)

    sys.path.insert(0, REPO_ROOT)
    from MARL_environment import make_env
    from consumer_step import solve_consumer_step

    # Create the environment without supplying a solution so it will generate
    # `D`, `PV` and `spot` (live solver mode). We'll compare the environment's
    # live-step output to the Python solver directly.
    env = make_env(None, n_prosumers=n, T=T)
    env.reset()

    reward_env = make_env(os.path.join(REPO_ROOT, "outputs", "opt_solution.json"), n_prosumers=n, T=T)
    reward_env.reset()
    total_reward = 0.0

    # iterate hours and inject optimal prices; for each hour call the Python
    # solver directly (expected) and then call env.step_with_action (actual)
    for t in range(T):
        action = prices[:, t]
        payload = {
            "prices": np.nan_to_num(np.asarray(action, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
            "D": np.nan_to_num(np.asarray(env.data["D"])[:, int(env.t)], nan=0.0, posinf=0.0, neginf=0.0).tolist(),
            "PV": np.nan_to_num(np.asarray(env.data["PV"])[:, int(env.t)], nan=0.0, posinf=0.0, neginf=0.0).tolist(),
            "soc": np.nan_to_num(np.asarray(env.soc, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
            "eta": float(env.data["eta"]),
            "E_max": np.nan_to_num(np.asarray(env.data["E_max"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
            "p_ch_max": np.nan_to_num(np.asarray(env.data["p_ch_max"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
            "p_dis_max": np.nan_to_num(np.asarray(env.data["p_dis_max"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
            "y_im": float(np.asarray(env.data["y_im"])[int(env.t)]),
            "y_ex": float(np.asarray(env.data["y_ex"])[int(env.t)]),
        }

        expected = solve_consumer_step(payload)

        # call the environment live-step
        actual = env.step_with_action(action.tolist())

        # compare all keys returned by the solver
        for key, exp_val in expected.items():
            got_val = actual.get(key)
            exp_arr = np.squeeze(np.array(exp_val))
            got_arr = np.squeeze(np.array(got_val))
            assert np.allclose(exp_arr, got_arr, atol=1e-6), f"Mismatch for {key} at t={t}: {exp_arr} vs {got_arr}"

        reward_env.step_with_action(action.tolist())
        total_reward += float(reward_env.rewards[reward_env.agents[0]])

    print("All hours matched between the Python solver and env live-step outputs for injected optimal prices.")
    assert np.isclose(total_reward, -float(sol["objective"]), atol=1e-5), f"Cumulative reward mismatch: {total_reward} vs {-float(sol['objective'])}"
    print(f"Aggregate reward matched the Julia objective with sign flipped: {total_reward}")

if __name__ == "__main__":
    test_env_matches_optimizer()
