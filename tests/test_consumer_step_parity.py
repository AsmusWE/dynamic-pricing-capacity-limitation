import json
import os
import subprocess
import sys
import tempfile

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def run_julia_optimizer():
    script = os.path.join(REPO_ROOT, "run_optimization.jl")
    subprocess.run(["julia", script], check=True, cwd=REPO_ROOT)


def load_solution():
    path = os.path.join(REPO_ROOT, "outputs", "opt_solution.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_price_matrix(x, T=24):
    arr = np.array(x)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected price matrix rank: {arr.ndim}")
    if arr.shape[1] == T:
        return arr
    if arr.shape[0] == T:
        return arr.T
    raise ValueError(f"Unexpected price matrix shape: {arr.shape}")


def build_payload(env, action, current_soc):
    t = int(env.t)
    return {
        "prices": np.nan_to_num(np.asarray(action, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
        "D": np.nan_to_num(np.asarray(env.data["D"])[:, t], nan=0.0, posinf=0.0, neginf=0.0).tolist(),
        "PV": np.nan_to_num(np.asarray(env.data["PV"])[:, t], nan=0.0, posinf=0.0, neginf=0.0).tolist(),
        "soc": np.nan_to_num(np.asarray(current_soc, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
        "eta": float(env.data["eta"]),
        "E_max": np.nan_to_num(np.asarray(env.data["E_max"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
        "p_ch_max": np.nan_to_num(np.asarray(env.data["p_ch_max"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
        "p_dis_max": np.nan_to_num(np.asarray(env.data["p_dis_max"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist(),
        "y_im": float(np.asarray(env.data["y_im"])[t]),
        "y_ex": float(np.asarray(env.data["y_ex"])[t]),
    }


def run_julia_consumer_step(payload):
    consumer_script = os.path.join(REPO_ROOT, "consumer_step.jl")
    if not os.path.exists(consumer_script):
        raise FileNotFoundError(f"Missing Julia consumer helper: {consumer_script}")

    with tempfile.TemporaryDirectory() as tmpdir:
        payload_path = os.path.join(tmpdir, "payload.json")
        output_path = os.path.join(tmpdir, "output.json")
        with open(payload_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        subprocess.run(["julia", consumer_script, payload_path, output_path], check=True, cwd=REPO_ROOT)
        with open(output_path, "r", encoding="utf-8") as handle:
            return json.load(handle)


def test_julia_and_python_consumer_step_match():
    run_julia_optimizer()
    solution = load_solution()
    prices = normalize_price_matrix(solution["x"], T=24)

    sys.path.insert(0, REPO_ROOT)
    from MARL_environment import make_env
    from consumer_step import solve_consumer_step

    env = make_env(None, n_prosumers=14, T=24)
    env.reset()

    current_soc = np.zeros(14, dtype=np.float32)
    for hour in range(24):
        action = prices[:, hour]
        payload = build_payload(env, action, current_soc)

        julia_result = run_julia_consumer_step(payload)
        python_result = solve_consumer_step(payload)

        for key in julia_result.keys():
            expected = np.squeeze(np.asarray(julia_result[key], dtype=np.float64))
            actual = np.squeeze(np.asarray(python_result[key], dtype=np.float64))
            assert np.allclose(expected, actual, atol=1e-6), f"Mismatch for {key} at hour {hour}: {expected} vs {actual}"

        current_soc = np.asarray(julia_result["next_soc"], dtype=np.float32)
        env.t += 1


if __name__ == "__main__":
    test_julia_and_python_consumer_step_match()