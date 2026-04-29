"""Test that data_loader.py produces values matching the Julia-generated opt_solution.json."""

import json, os, sys
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

def load_solution():
    with open(os.path.join(REPO_ROOT, "outputs", "opt_solution.json"), "r") as f:
        return json.load(f)

def _normalize(arr, n, t):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.shape == (n, t): return arr
    if arr.shape == (t, n): return arr.T
    raise ValueError(f"Bad shape {arr.shape}")

def _check(name, exp, act, n=14, t=24, tol=1e-4):
    e, a = _normalize(exp, n, t), _normalize(act, n, t)
    assert np.allclose(e, a, atol=tol), f"{name}: max diff {np.abs(e-a).max():.6f}"

def test_demand():
    sys.path.insert(0, REPO_ROOT)
    from data_loader import load_data
    s, d = load_solution(), load_data(os.path.join(REPO_ROOT,"Data"), 14, 1234)
    _check("Demand", s["D"], d["D"])

def test_pv():
    sys.path.insert(0, REPO_ROOT)
    from data_loader import load_data
    s, d = load_solution(), load_data(os.path.join(REPO_ROOT,"Data"), 14, 1234)
    _check("PV", s["PV"], d["PV"])

def test_spot():
    sys.path.insert(0, REPO_ROOT)
    from data_loader import load_data
    s, d = load_solution(), load_data(os.path.join(REPO_ROOT,"Data"), 14, 1234)
    _check("spot", s["spot"], d["spot"], n=1, t=24)

def test_y_im():
    sys.path.insert(0, REPO_ROOT)
    from data_loader import load_data
    s, d = load_solution(), load_data(os.path.join(REPO_ROOT,"Data"), 14, 1234)
    _check("y_im", s["y_im"], d["y_im"], n=1, t=24)

def test_y_ex():
    sys.path.insert(0, REPO_ROOT)
    from data_loader import load_data
    s, d = load_solution(), load_data(os.path.join(REPO_ROOT,"Data"), 14, 1234)
    _check("y_ex", s["y_ex"], d["y_ex"], n=1, t=24)

def test_battery():
    sys.path.insert(0, REPO_ROOT)
    from data_loader import load_data
    d = load_data(os.path.join(REPO_ROOT,"Data"), 14, 1234)
    exp = np.array([1,2,0,3,1,1,3,3,1,2,2,2,0,0], dtype=np.float32)
    act = np.asarray(d["E_max"], dtype=np.float32) / 5.0
    assert np.allclose(exp, act), f"max_cap: {act}"
    assert np.allclose(exp*5, np.asarray(d["E_max"]))
    assert np.allclose(exp*5/2, np.asarray(d["p_ch_max"]))

if __name__ == "__main__":
    test_demand(); test_pv(); test_spot(); test_y_im(); test_y_ex(); test_battery()
    print("All data loader tests passed!")
