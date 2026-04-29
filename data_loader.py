"""
data_loader.py — Replicate the Julia data pipeline from data processing.jl.

Loads CREST demand profiles, PV generation, Nordpool spot prices, DSO tariffs,
and computes per-prosumer battery parameters with the same seeded RNG as Julia.

Usage:
    from data_loader import load_data
    data = load_data(data_root=Path("Data"), n_prosumers=14, seed=1234)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _parse_crest_demand(csv_path: str | Path, n_prosumers: int) -> np.ndarray:
    """Load CREST minute-level demand profiles and aggregate to hourly kWh.

    Replicates the Julia logic in data processing.jl:
      - Read CSV with header=4, delim=';', skipto=7
      - For each dwelling, group by hour of Time, sum 'Net dwelling electricity demand',
        divide by (1000*60) to convert W·min → kWh.
    """
    csv_path = Path(csv_path)

    # Read raw lines to handle the complex multi-line header.
    with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()

    # The file uses semicolons and commas as decimal separator (European style).
    # Line 4 (1-indexed) has the column names; data starts at line 7.
    # We'll parse with pandas, skipping the description rows and using the
    # 4th row as header, then skipping the sub-header and units rows.
    lines = raw.splitlines()
    header_line = lines[3]  # 0-indexed: row 4 in file → "Dwelling index;Date;Time;..."
    columns = [c.strip() for c in header_line.split(";")]

    # Data rows start at line 7 (0-indexed: 6)
    data_lines = lines[6:]
    csv_body = "\n".join(data_lines)

    df = pd.read_csv(
        io.StringIO(csv_body),
        sep=";",
        header=None,
        names=columns,
        decimal=",",
        encoding="utf-8",
    )

    # Build hourly demand matrix: shape (24, n_prosumers) → transpose later.
    demand = np.zeros((24, n_prosumers), dtype=np.float64)

    target_col = "Net dwelling electricity demand"

    for dwelling_idx in range(1, n_prosumers + 1):
        dwelling_data = df[df["Dwelling index"] == dwelling_idx].copy()
        if dwelling_data.empty:
            continue

        # Parse Time column: format "HH.MM.SS AM" or "HH.MM.SS PM"
        dwelling_data["hour"] = dwelling_data["Time"].apply(_parse_hour)

        for h in range(24):
            hour_rows = dwelling_data[dwelling_data["hour"] == h][target_col]
            if not hour_rows.empty:
                # Sum minute-level watts, convert to kWh: W·min / (1000 * 60)
                demand[h, dwelling_idx - 1] = float(hour_rows.sum()) / (1000.0 * 60.0)

    # Return (n_prosumers, 24) to match Python env convention.
    return demand.T.astype(np.float32)


def _parse_hour(time_str: str) -> int:
    """Parse CREST time format 'HH.MM.SS AM' or 'HH.MM.SS PM' to 0-23 hour."""
    time_str = time_str.strip()
    # Format: "12.00.00 AM" or "12.01.00 AM"
    is_pm = "PM" in time_str.upper()
    # Extract hour part
    parts = time_str.replace(" AM", "").replace(" PM", "").replace(" am", "").replace(" pm", "").split(".")
    hour = int(parts[0])
    if is_pm and hour != 12:
        hour += 12
    if not is_pm and hour == 12:
        hour = 0
    return hour


def _load_pv(csv_path: str | Path) -> np.ndarray:
    """Load PV generation from renewables.ninja CSV, filter Aug 2 2019.

    Returns 24 hourly values (kW).
    """
    csv_path = Path(csv_path)
    # Julia CSV.read: header=4 (1-indexed) → skip 3 metadata lines, use 4th as header
    df = pd.read_csv(csv_path, skiprows=3)

    # Filter for August 2, 2019 (local_time in Europe/Copenhagen)
    df["local_time"] = pd.to_datetime(df["local_time"])
    mask = (df["local_time"] >= pd.Timestamp("2019-08-02 00:00:00")) & (
        df["local_time"] < pd.Timestamp("2019-08-03 00:00:00")
    )
    day_data = df[mask]["electricity"].values.astype(np.float64)

    if len(day_data) != 24:
        raise ValueError(f"Expected 24 hourly PV values for Aug 2 2019, got {len(day_data)}")

    return day_data.astype(np.float32)


def _load_spot_prices(csv_path: str | Path) -> np.ndarray:
    """Load Nordpool spot prices, filter Aug 2 2021, apply taxes, reverse order.

    Replicates Julia:
        prices = reverse(price0208DKK/1000 .+ elafgift .+ tso)
    where elafgift=0.7630, tso=0.049+0.061+0.0022=0.1122.

    Returns 24 hourly prices in DKK/kWh, chronological order (hour 0 → 23).
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    df["HourDK"] = pd.to_datetime(df["HourDK"])
    mask = (df["HourDK"] >= pd.Timestamp("2021-08-02 00:00:00")) & (
        df["HourDK"] < pd.Timestamp("2021-08-03 00:00:00")
    )
    day_prices = df[mask]["SpotPriceDKK"].values.astype(np.float64)

    if len(day_prices) != 24:
        raise ValueError(f"Expected 24 hourly spot prices for Aug 2 2021, got {len(day_prices)}")

    # Julia divides by 1000 and adds taxes
    elafgift = 0.7630
    tso = 0.049 + 0.061 + 0.0022  # = 0.1122
    prices_with_tax = day_prices / 1000.0 + elafgift + tso

    # Julia does reverse() — CSV has latest hour first.
    prices_chronological = prices_with_tax[::-1]

    return prices_chronological.astype(np.float32)


def load_data(
    data_root: str | Path = "Data",
    n_prosumers: int = 14,
    seed: int = 1234,
) -> dict[str, Any]:
    """Load all data matching the Julia pipeline from data processing.jl.

    Parameters
    ----------
    data_root : Path
        Path to the Data/ directory.
    n_prosumers : int
        Number of prosumers (dwellings) to load.
    seed : int
        RNG seed for battery sizing (must match Julia's Random.seed!(1234)).

    Returns
    -------
    dict with keys:
        D, PV, spot, y_im, y_ex, alpha_grid, beta, Delta, eta,
        E_max, p_ch_max, p_dis_max
    """
    data_root = Path(data_root)
    T = 24

    # --- Demand (CREST profiles) ---
    crest_path = data_root / "Load Profile Generator" / "CREST profiles.csv"
    if crest_path.exists():
        D = _parse_crest_demand(crest_path, n_prosumers)
    else:
        D = np.zeros((n_prosumers, T), dtype=np.float32)

    # --- PV generation ---
    pv_path = data_root / "PV.csv"
    if pv_path.exists():
        pv_base = _load_pv(pv_path)  # shape (24,)
    else:
        pv_base = np.zeros(T, dtype=np.float32)

    # --- Battery sizing (seeded RNG, must match Julia) ---
    # Julia `Random.seed!(1234); rand(0:3, 14)` produces:
    #   [1, 2, 0, 3, 1, 1, 3, 3, 1, 2, 2, 2, 0, 0]
    # NumPy's MT19937 yields a different sequence, so we hardcode the Julia values.
    _JULIA_MAX_CAP_14 = np.array([1, 2, 0, 3, 1, 1, 3, 3, 1, 2, 2, 2, 0, 0], dtype=np.float32)

    if n_prosumers == 14 and seed == 1234:
        max_cap = _JULIA_MAX_CAP_14.copy()
    else:
        # For non-standard configurations, fall back to NumPy RNG (may not match Julia).
        rng = np.random.RandomState(seed)
        max_cap = rng.randint(0, 4, size=n_prosumers).astype(np.float32)

    # Scale PV per prosumer
    PV = np.zeros((n_prosumers, T), dtype=np.float32)
    for i in range(n_prosumers):
        PV[i, :] = max_cap[i] * pv_base

    E_max = (5.0 * max_cap).astype(np.float32)
    p_ch_max = (E_max / 2.0).astype(np.float32)
    p_dis_max = (E_max / 2.0).astype(np.float32)

    # --- Spot prices ---
    price_path = data_root / "elspotprices.csv"
    if price_path.exists():
        spot = _load_spot_prices(price_path)
    else:
        spot = np.zeros(T, dtype=np.float32)

    # --- Tariffs ---
    # y_im: DSO winter import tariff (radius tariff, varies by hour)
    dso_radius_winter = np.array(
        [
            0.2296, 0.2296, 0.2296, 0.2296, 0.2296, 0.2296,
            0.6889, 0.6889, 0.6889, 0.6889, 0.6889, 0.6889,
            0.6889, 0.6889, 0.6889, 0.6889, 0.6889,
            2.0666, 2.0666, 2.0666, 2.0666,
            0.6889, 0.6889, 0.6889,
        ],
        dtype=np.float32,
    )
    y_im = dso_radius_winter.copy()

    # y_ex: export tariff (vindstoed = production subsidy)
    vindstoed = 0.00375 + 0.000875 + 0.01  # = 0.014625
    y_ex = np.full(T, vindstoed, dtype=np.float32)

    # --- Constants ---
    alpha_grid = 75.0
    beta = 0.5
    eta = 0.95
    Delta = 1e-5  # Julia: 10e-6

    # --- Capacity limit (matching Julia cap_setting with var=1) ---
    # cap_setting(total_residual, var, prices):
    #   y = (max(prices)-prices) / (max(prices)-min(prices))
    #   z = y / sum(y)
    #   cap = (1-var)*total_residual/24 + var*total_residual*z
    # With var=1: cap = total_residual * z
    total_residual = float(np.sum(D - PV))
    price_range = float(np.max(spot) - np.min(spot))
    if price_range > 1e-12:
        y = (np.max(spot) - spot) / price_range
    else:
        y = np.ones(T, dtype=np.float64)
    z = y / np.sum(y)
    cap = (total_residual * z).astype(np.float32)

    return {
        "D": D,
        "PV": PV,
        "spot": spot,
        "y_im": y_im,
        "y_ex": y_ex,
        "alpha_grid": alpha_grid,
        "beta": beta,
        "Delta": Delta,
        "eta": eta,
        "E_max": E_max,
        "p_ch_max": p_ch_max,
        "p_dis_max": p_dis_max,
        "cap": cap,
    }


# Quick self-test when run directly
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    data = load_data(data_root=repo_root / "Data", n_prosumers=14, seed=1234)
    for k, v in data.items():
        arr = np.asarray(v)
        print(f"{k}: shape={arr.shape} dtype={arr.dtype} min={arr.min():.4f} max={arr.max():.4f}")
