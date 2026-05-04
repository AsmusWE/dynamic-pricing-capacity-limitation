"""
extract_optimal_actions.jl — Extract LP-optimal battery actions for the Aug-2 reference day.

Runs the community-welfare LP (beta=0.5, cap variation=1, Aug 2 data, line capacities
enabled) and exports the per-prosumer, per-hour charge/discharge actions to
  Data/optimal_actions.csv

This file only needs to be run once.  The Python training script (marl_main.py) will
load the CSV automatically to compute an LP-optimal benchmark during training.

Usage:
    julia extract_optimal_actions.jl

Requirements: Julia with JuMP, Gurobi, CSV, DataFrames installed and a valid Gurobi licence.
"""

using LaTeXStrings, Statistics, Gurobi, CSV, DataFrames

root = dirname(@__FILE__)
data_path = joinpath(root, "Data/")

include("model.jl")
include("data processing.jl")

const GRB_ENV = Gurobi.Env(Dict{String,Any}("OutputFlag" => 1, "MIPGap" => 1e-4))

N_consumer = 14
grid = "feeder15"

println("Loading data...")
data, node, line, ind_cost, tot_cost, ind_profile, tot_profile =
    data_processing(data_path, grid, N_consumer, GRB_ENV)

residual       = sum(eachcol(data["D"] .- data["PV"]))
total_residual = sum(sum(eachcol(data["D"] .- data["PV"])))

# Reference scenario: cap variation = 1 (fully price-responsive capacity limit)
cap_ref = cap_setting(total_residual, 1, data["spot"])

println("Computing individual benchmarks...")
community_benchmark, individual_benchmark = benchmarks(ind_cost, ind_profile, tot_profile, cap_ref)

# Set initial SoC to 0 (empty batteries at start)
if haskey(data, "SoC_initial")
    data["SoC_initial"] .= 0.0
end

println("Solving LP (beta=0.5, cap=1, Aug-2, with line capacities, initial SoC=0)...")
sol = dynamic_pricing(data, node, line, individual_benchmark, 0.5, cap_ref, "none", true, GRB_ENV)

if !has_values(sol)
    error("LP solver did not find a primal solution (status: $(termination_status(sol)))")
end

# Extract battery actions — shape: 14 prosumers × 24 hours (Julia 1-indexed)
p_ch_mat  = Array(value.(sol[:pᶜʰ]))   # 14×24
p_dis_mat = Array(value.(sol[:pᵈⁱˢ]))  # 14×24

p_ch_max  = data["pᶜʰᵐᵃˣ"]   # 14-element vector
p_dis_max = data["pᵈⁱˢᵐᵃˣ"]  # 14-element vector
E_max     = data["Eᵐᵃˣ"]      # 14-element vector

println("Building output DataFrame...")
rows = []
for i in 1:N_consumer
    for t in 1:24
        pch  = p_ch_mat[i, t]
        pdis = p_dis_mat[i, t]

        # Normalise to [-1, 1]: +1 = full charge, -1 = full discharge
        if E_max[i] == 0.0
            action = 0.0
        else
            ch_norm  = p_ch_max[i]  > 0.0 ? pch  / p_ch_max[i]  : 0.0
            dis_norm = p_dis_max[i] > 0.0 ? pdis / p_dis_max[i] : 0.0
            action   = clamp(ch_norm - dis_norm, -1.0, 1.0)
        end

        push!(rows, (
            prosumer = i - 1,   # 0-indexed to match Python
            hour     = t - 1,   # 0-indexed to match Python
            p_ch     = pch,
            p_dis    = pdis,
            action   = action,
        ))
    end
end

df = DataFrame(rows)
out_path = joinpath(root, "Data", "optimal_actions.csv")
CSV.write(out_path, df)
println("Saved $(nrow(df)) rows to: $out_path")

# Quick sanity print
println("\nSample actions (prosumer 0, hours 0-5):")
println(filter(r -> r.prosumer == 0 && r.hour < 6, df))
