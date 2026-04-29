"""Consumer-step solver matching the KKT-based lower-level formulation in model.jl.

Changes from the previous version:
- DC-side battery variables: power balance uses eta*p_dis - p_ch/eta,
  battery balance uses p_ch - p_dis (no eta).
- Cyclic battery constraint: e[0] = e[23] + p_ch[0] - p_dis[0].
- Objective: x_it * (p+ - p-) + VOLL * d_shed  (no tariffs in lower level).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import gurobipy as gp
import numpy as np


def _as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _as_price_matrix(values: Any, n_rows: int, horizon: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        if array.size != n_rows * horizon:
            raise ValueError(f"Expected {n_rows * horizon} price values, got {array.size}")
        return array.reshape(n_rows, horizon)
    if array.ndim == 2:
        if array.shape == (n_rows, horizon):
            return array
        if array.shape == (horizon, n_rows):
            return array.T
    raise ValueError(f"Unexpected price matrix shape: {array.shape}")


def _as_matrix(values: Any, n_rows: int, horizon: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        if array.size != n_rows * horizon:
            raise ValueError(f"Expected {n_rows * horizon} entries for {name}, got {array.size}")
        return array.reshape(n_rows, horizon)
    if array.ndim == 2:
        if array.shape == (n_rows, horizon):
            return array
        if array.shape == (horizon, n_rows):
            return array.T
    raise ValueError(f"Unexpected {name} shape: {array.shape}")


def solve_consumer_step(payload: dict[str, Any]) -> dict[str, list[float]]:
    prices_raw = payload["prices"]
    demand_raw = payload["D"]
    pv_raw = payload["PV"]
    soc = _as_float_array(payload["soc"])
    eta = float(payload["eta"])
    e_max = _as_float_array(payload["E_max"])
    p_ch_max = _as_float_array(payload["p_ch_max"])
    p_dis_max = _as_float_array(payload["p_dis_max"])
    horizon = int(np.asarray(payload.get("y_im", [0]*24)).shape[-1]) if np.asarray(payload.get("y_im", [0]*24)).ndim > 0 else 24
    demand = _as_matrix(demand_raw, soc.shape[0], horizon, "D")
    pv = _as_matrix(pv_raw, soc.shape[0], horizon, "PV")
    n_prosumers = soc.shape[0]
    prices = _as_price_matrix(prices_raw, n_prosumers, horizon)

    expected_lengths = {
        "D": demand.shape[0], "PV": pv.shape[0], "soc": soc.shape[0],
        "E_max": e_max.shape[0], "p_ch_max": p_ch_max.shape[0],
        "p_dis_max": p_dis_max.shape[0],
    }
    if len({n_prosumers, *expected_lengths.values()}) != 1:
        raise ValueError(f"Mismatched payload lengths: {n_prosumers=} {expected_lengths=}")

    results: dict[str, list[float]] = {
        "p_im": [], "p_ex": [], "p_pen": [], "p_plus": [], "p_minus": [],
        "p_ch": [], "p_dis": [], "e": [], "d_shed": [], "f_p": [], "f_q": [],
        "next_soc": [],
    }

    VOLL = 1.25 * 75.0

    for idx in range(n_prosumers):
        model = gp.Model()
        model.Params.OutputFlag = 0

        # --- Variables (DC-side battery formulation matching KKT) ---
        p_plus_v  = model.addVars(horizon, lb=0.0, name="p_plus")
        p_minus_v = model.addVars(horizon, lb=0.0, name="p_minus")
        # KKT uses DC-side p_ch/p_dis: power balance has eta*p_dis - p_ch/eta
        # DC-side bounds: p_ch_dc_max = eta * p_ch_max_ac = eta * (E_max/2)
        # But the centralized model bounds p_ch <= E_max/2 directly on DC side
        p_ch_max_dc = float(e_max[idx]) / 2.0
        p_dis_max_dc = float(e_max[idx]) / 2.0
        p_ch_v  = model.addVars(horizon, lb=0.0, ub=p_ch_max_dc, name="p_ch")
        p_dis_v = model.addVars(horizon, lb=0.0, ub=p_dis_max_dc, name="p_dis")
        e_v     = model.addVars(horizon, lb=0.0, ub=float(e_max[idx]), name="e")
        d_shed_v = model.addVars(horizon, lb=0.0, name="d_shed")
        q_plus_v  = model.addVars(horizon, lb=0.0, name="q_plus")
        q_minus_v = model.addVars(horizon, lb=0.0, name="q_minus")

        # --- Constraints ---
        for t in range(horizon):
            # Power balance (DC-side): p+ - p- + PV - D + d_shed + eta*p_dis - p_ch/eta == 0
            model.addConstr(
                p_plus_v[t] - p_minus_v[t]
                + float(pv[idx, t]) - float(demand[idx, t])
                + d_shed_v[t]
                + eta * p_dis_v[t] - p_ch_v[t] / eta
                == 0,
                name=f"power_balance_{t}",
            )

            # Battery balance (DC-side, no eta)
            if t == 0:
                # Cyclic: e[0] = e[23] + p_ch[0] - p_dis[0]
                model.addConstr(
                    e_v[0] == e_v[horizon - 1] + p_ch_v[0] - p_dis_v[0],
                    name="battery_balance_0_cyclic",
                )
            else:
                model.addConstr(
                    e_v[t] == e_v[t - 1] + p_ch_v[t] - p_dis_v[t],
                    name=f"battery_balance_{t}",
                )

            # Reactive power: q = sigma * p  (sigma = tan_phi = 0.5)
            model.addConstr(q_plus_v[t] == 0.5 * p_plus_v[t], name=f"reactive_plus_{t}")
            model.addConstr(q_minus_v[t] == 0.5 * p_minus_v[t], name=f"reactive_minus_{t}")

        # --- Objective: min Σ [ x·(p⁺ - p⁻) + VOLL·d_shed ] ---
        # The optimal KKT prices satisfy x_high/x_low = 1/eta^2 exactly,
        # making battery cycling zero-profit.  The LP may pick any combination
        # of battery usage — all are equally optimal for the consumer.
        # The RL agent must learn prices with a larger spread to make battery
        # cycling strictly profitable, enabling capacity management.
        obj = gp.quicksum(
            float(prices[idx, t]) * (p_plus_v[t] - p_minus_v[t])
            + VOLL * d_shed_v[t]
            for t in range(horizon)
        )
        model.setObjective(obj, gp.GRB.MINIMIZE)
        model.optimize()

        if model.SolCount == 0:
            p_plus_val  = [0.0] * horizon
            p_minus_val = [0.0] * horizon
            p_ch_val    = [0.0] * horizon
            p_dis_val   = [0.0] * horizon
            e_val       = [0.0] * horizon
            d_shed_val  = [max(0.0, float(demand[idx, t] - pv[idx, t])) for t in range(horizon)]
        else:
            p_plus_val  = [float(p_plus_v[t].X) for t in range(horizon)]
            p_minus_val = [float(p_minus_v[t].X) for t in range(horizon)]
            p_ch_val    = [float(p_ch_v[t].X) for t in range(horizon)]
            p_dis_val   = [float(p_dis_v[t].X) for t in range(horizon)]
            e_val       = [float(e_v[t].X) for t in range(horizon)]
            d_shed_val  = [float(d_shed_v[t].X) for t in range(horizon)]

        results["p_im"].append(list(p_plus_val))
        results["p_ex"].append(list(p_minus_val))
        results["p_pen"].append([0.0] * horizon)
        results["p_plus"].append(list(p_plus_val))
        results["p_minus"].append(list(p_minus_val))
        results["p_ch"].append(list(p_ch_val))
        results["p_dis"].append(list(p_dis_val))
        results["e"].append(list(e_val))
        results["d_shed"].append(list(d_shed_val))
        results["f_p"].append([pp - pm for pp, pm in zip(p_plus_val, p_minus_val)])
        results["f_q"].append([0.5 * (pp - pm) for pp, pm in zip(p_plus_val, p_minus_val)])
        results["next_soc"].append(float(e_val[-1]))

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
