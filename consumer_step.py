from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import gurobipy as gp
import numpy as np


def _as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def solve_consumer_step(payload: dict[str, Any]) -> dict[str, list[float]]:
    prices = _as_float_array(payload["prices"])
    demand = _as_float_array(payload["D"])
    pv = _as_float_array(payload["PV"])
    soc = _as_float_array(payload["soc"])
    eta = float(payload["eta"])
    e_max = _as_float_array(payload["E_max"])
    p_ch_max = _as_float_array(payload["p_ch_max"])
    p_dis_max = _as_float_array(payload["p_dis_max"])
    y_im = float(payload["y_im"])
    y_ex = float(payload["y_ex"])

    n_prosumers = prices.shape[0]
    expected_lengths = {
        "D": demand.shape[0],
        "PV": pv.shape[0],
        "soc": soc.shape[0],
        "E_max": e_max.shape[0],
        "p_ch_max": p_ch_max.shape[0],
        "p_dis_max": p_dis_max.shape[0],
    }
    if len({n_prosumers, *expected_lengths.values()}) != 1:
        raise ValueError(f"Mismatched payload lengths: {n_prosumers=} {expected_lengths=}")

    results: dict[str, list[float]] = {
        "p_plus": [],
        "p_minus": [],
        "p_ch": [],
        "p_dis": [],
        "e": [],
        "d_shed": [],
        "f_p": [],
        "f_q": [],
        "next_soc": [],
    }

    for idx in range(n_prosumers):
        model = gp.Model()
        model.Params.OutputFlag = 0

        p_plus = model.addVar(lb=0.0, name="p_plus")
        p_minus = model.addVar(lb=0.0, name="p_minus")
        p_ch = model.addVar(lb=0.0, ub=float(p_ch_max[idx]), name="p_ch")
        p_dis = model.addVar(lb=0.0, ub=float(p_dis_max[idx]), name="p_dis")
        e = model.addVar(lb=0.0, ub=float(e_max[idx]), name="e")
        d_shed = model.addVar(lb=0.0, name="d_shed")
        q_plus = model.addVar(lb=0.0, name="q_plus")
        q_minus = model.addVar(lb=0.0, name="q_minus")

        model.setObjective(
            (float(prices[idx]) + y_im) * p_plus
            - (float(prices[idx]) - y_ex) * p_minus
            + 1.25 * 75.0 * d_shed,
            gp.GRB.MINIMIZE,
        )
        model.addConstr(p_plus - p_minus + float(pv[idx]) - float(demand[idx]) + p_dis - p_ch + d_shed == 0, name="power_balance")
        model.addConstr(e == float(soc[idx]) + eta * p_ch - p_dis / eta, name="battery_balance")
        model.addConstr(q_plus == 0.5 * p_plus, name="reactive_plus")
        model.addConstr(q_minus == 0.5 * p_minus, name="reactive_minus")

        model.optimize()

        if model.SolCount == 0:
            p_plus_value = 0.0
            p_minus_value = 0.0
            p_ch_value = 0.0
            p_dis_value = 0.0
            e_value = float(soc[idx])
            d_shed_value = float(demand[idx] - pv[idx])
        else:
            p_plus_value = float(p_plus.X)
            p_minus_value = float(p_minus.X)
            p_ch_value = float(p_ch.X)
            p_dis_value = float(p_dis.X)
            e_value = float(e.X)
            d_shed_value = float(d_shed.X)

        results["p_plus"].append(p_plus_value)
        results["p_minus"].append(p_minus_value)
        results["p_ch"].append(p_ch_value)
        results["p_dis"].append(p_dis_value)
        results["e"].append(e_value)
        results["d_shed"].append(d_shed_value)
        results["f_p"].append(p_plus_value - p_minus_value)
        results["f_q"].append(0.5 * (p_plus_value - p_minus_value))
        results["next_soc"].append(e_value)

    return results


def run_cli(payload_path: str, output_path: str) -> None:
    payload_file = Path(payload_path)
    output_file = Path(output_path)
    with payload_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = solve_consumer_step(payload)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(results, handle)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("Usage: python consumer_step.py <payload.json> <output.json>", file=sys.stderr)
        return 2
    run_cli(args[0], args[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())