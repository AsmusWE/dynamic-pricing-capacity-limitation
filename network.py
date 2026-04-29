"""
network.py — Radial distribution network (LinDistFlow) for power flow calculation.

Loads feeder topology from CSV and computes downstream power flows given
per-prosumer net injections (p_plus - p_minus). Checks line limit violations
and returns penalties matching the formulation in model.jl.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class FeederNetwork:
    """Represents a radial distribution feeder for LinDistFlow calculations."""

    def __init__(self, data_root: str | Path, feeder_name: str = "feeder15"):
        data_root = Path(data_root)
        feeder_dir = data_root / "testcase" / feeder_name

        # Load CSV data
        nodes_df = pd.read_csv(feeder_dir / "nodes.csv")
        lines_df = pd.read_csv(feeder_dir / "lines.csv")

        # Try to load generator data (may not exist)
        gen_path = feeder_dir / "generators.csv"
        gen_df = pd.read_csv(gen_path) if gen_path.exists() else None

        self.n_nodes = len(nodes_df)
        self.n_lines = len(lines_df)
        self.n_prosumers = self.n_nodes - 1  # node 0 is substation

        # --- Build tree structure ---
        # parent[i] = parent node index of node i (0 = substation)
        self.parent = np.zeros(self.n_nodes, dtype=np.int32)
        # children[i] = list of child node indices
        self.children: list[list[int]] = [[] for _ in range(self.n_nodes)]
        # line index for each node (line connecting parent→node)
        self.line_idx = np.zeros(self.n_nodes, dtype=np.int32)

        # line_r[i], line_x[i] = resistance/reactance of line TO node i
        self.line_r = np.zeros(self.n_nodes, dtype=np.float64)
        self.line_x = np.zeros(self.n_nodes, dtype=np.float64)
        self.line_s_max = np.zeros(self.n_nodes, dtype=np.float64)

        for _, row in lines_df.iterrows():
            f = int(row["node_f"])
            t = int(row["node_t"])
            idx = int(row["index"])
            self.parent[t] = f
            self.children[f].append(t)
            self.line_idx[t] = idx
            self.line_r[t] = float(row["r"])
            self.line_x[t] = float(row["x"])
            self.line_s_max[t] = float(row["s_max"])

        # --- Node parameters ---
        # tan_phi: reactive-to-active power ratio (from generator or default 0)
        self.tan_phi = np.zeros(self.n_nodes, dtype=np.float64)
        if gen_df is not None:
            for _, row in gen_df.iterrows():
                node = int(row["node"])
                if node > 0:  # skip substation generator
                    self.tan_phi[node] = 0.5  # hardcoded in data_manager.jl

        # v_max, v_min per node (voltage limits)
        self.v_max = nodes_df["v_max"].values.astype(np.float64)
        self.v_min = nodes_df["v_min"].values.astype(np.float64)

        # --- Precompute node order for bottom-up flow computation ---
        # Post-order traversal: children before parents
        self._post_order = self._compute_post_order()

    def _compute_post_order(self) -> list[int]:
        """Return nodes in post-order (children before parents)."""
        order = []

        def dfs(node: int):
            for child in self.children[node]:
                dfs(child)
            if node > 0:  # skip substation
                order.append(node)

        dfs(0)
        return order

    def compute_flows(
        self, p_plus: np.ndarray, p_minus: np.ndarray, S_base: float
    ) -> dict[str, np.ndarray]:
        """Compute downstream flows given per-prosumer net injections.

        Parameters
        ----------
        p_plus : (n_prosumers, T) array — per-prosumer import [kW]
        p_minus : (n_prosumers, T) array — per-prosumer export [kW]
        S_base : float — base power for per-unit conversion [kW]

        Returns
        -------
        dict with:
            f_p : (n_nodes, T) — active power flow on line TO each node [kW]
            f_q : (n_nodes, T) — reactive power flow on line TO each node [kVAr]
            u   : (n_nodes, T) — squared voltage at each node [p.u.]
            line_violation : (n_nodes, T) — max(0, S_line - s_max) [p.u.]
            voltage_violation : (n_nodes, T) — max(0, v_min - u) + max(0, u - v_max)
        """
        T = p_plus.shape[1]
        n = self.n_nodes

        # Net injections at each prosumer node (1-indexed)
        p_net = np.zeros((n, T), dtype=np.float64)
        q_net = np.zeros((n, T), dtype=np.float64)

        for i in range(self.n_prosumers):
            node = i + 1  # prosumer i corresponds to node i+1
            p_net[node] = p_plus[i] - p_minus[i]
            q_net[node] = self.tan_phi[node] * p_net[node]

        # Flows (positive = away from substation, i.e., from parent to child)
        f_p = np.zeros((n, T), dtype=np.float64)
        f_q = np.zeros((n, T), dtype=np.float64)

        # Bottom-up: for each node in post-order, sum net injection + children flows
        for node in self._post_order:
            child_flow_p = sum(f_p[child] for child in self.children[node])
            child_flow_q = sum(f_q[child] for child in self.children[node])
            f_p[node] = p_net[node] + child_flow_p
            f_q[node] = q_net[node] + child_flow_q

        # Voltage computation (top-down from substation)
        u = np.ones((n, T), dtype=np.float64)  # u[0] = 1.0 p.u.

        def compute_voltage(node: int):
            for child in self.children[node]:
                r = self.line_r[child]
                x = self.line_x[child]
                # u[child] = u[node] - 2*((f_p/S_base)*r + (f_q/S_base)*x)
                u[child] = (
                    u[node]
                    - 2.0 * ((f_p[child] / S_base) * r + (f_q[child] / S_base) * x)
                )
                compute_voltage(child)

        compute_voltage(0)

        # --- Line limit violations ---
        # S_line = sqrt((f_p/S_base)^2 + (f_q/S_base)^2) in per-unit
        S_line = np.sqrt((f_p / S_base) ** 2 + (f_q / S_base) ** 2)
        line_violation = np.maximum(0.0, S_line - self.line_s_max.reshape(-1, 1))

        # --- Voltage violations ---
        v_min_arr = self.v_min.reshape(-1, 1)
        v_max_arr = self.v_max.reshape(-1, 1)
        voltage_violation = np.maximum(0.0, v_min_arr - u) + np.maximum(
            0.0, u - v_max_arr
        )

        return {
            "f_p": f_p,
            "f_q": f_q,
            "u": u,
            "line_violation": line_violation,
            "voltage_violation": voltage_violation,
        }


def compute_S_base(data_root: str | Path, feeder_name: str, demand: np.ndarray) -> float:
    """Compute S_base matching Julia's data_processing.jl calculation.

    S_base = sqrt(max(D)^2 + (0.5*max(D))^2) / S_max
    where S_max = max over nodes of sqrt(d_P^2 + d_Q^2).
    """
    feeder_dir = Path(data_root) / "testcase" / feeder_name
    nodes_df = pd.read_csv(feeder_dir / "nodes.csv")

    # S_max = max apparent power of node demand
    d_p = nodes_df["d_P"].values.astype(np.float64)
    d_q = nodes_df["d_Q"].values.astype(np.float64)
    S_node = np.sqrt(d_p**2 + d_q**2)
    S_max = float(np.max(S_node))

    # Max demand from data (note: demand shape is (n_prosumers, T))
    max_demand = float(np.max(demand))

    S_base = np.sqrt(max_demand**2 + (0.5 * max_demand) ** 2) / S_max
    return S_base
