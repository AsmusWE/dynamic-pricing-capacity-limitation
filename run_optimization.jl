import Pkg

# ensure the correct project is active so dependencies (JSON, JuMP, Gurobi) are available
root = dirname(@__FILE__)
proj_path = joinpath(root, "DP_CC_OPF")
if isdir(proj_path) && isfile(joinpath(proj_path, "Project.toml"))
    Pkg.activate(proj_path)
else
    Pkg.activate(root)
end
# instantiate if packages are missing (safe if already instantiated)
try
    Pkg.instantiate()
catch err
    @warn "Pkg.instantiate failed" err
end

using JSON, Gurobi
using JuMP

data_path = joinpath(root, "Data/")

include(joinpath(root, "model.jl"))
include(joinpath(root, "data processing.jl"))

const GRB_ENV = Gurobi.Env(Dict{String,Any}("OutputFlag"=>0, "MIPGap"=>1e-4))

println("Preparing data and running optimizer...")
N_consumer = 14
grid = "feeder15"
data, node, line, ind_cost, tot_cost, ind_profile, tot_profile = data_processing(data_path,grid,N_consumer,GRB_ENV)

residual = sum(eachcol(data["D"] .- data["PV"]))
total_residual = sum(sum(eachcol(data["D"] .- data["PV"])))

cap_14 = cap_setting(total_residual,1,data["spot"])
community_benchmark, individual_benchmark = benchmarks(ind_cost,ind_profile,tot_profile, cap_14)
b = 0.5

model = dynamic_pricing(data,node,line,individual_benchmark,b,cap_14,"none",true,GRB_ENV)

if !has_values(model)
    println("No primal solution available; status=", termination_status(model))
    exit(1)
end

println("Collecting variables and saving JSON...")

canonicalize_2d(v) = permutedims(Array(value.(v)))

function canonicalize_data_2d(v)
    arr = Array(v)
    return ndims(arr) == 2 ? permutedims(arr) : arr
end

sol = Dict()
sol["objective"] = objective_value(model)
sol["x"] = canonicalize_2d(model[:x])  # shape: I x T
sol["p_im"] = canonicalize_2d(model[:pⁱᵐ])
sol["p_ex"] = canonicalize_2d(model[:pᵉˣ])
sol["p_pen"] = collect(value.(model[:pᵖᵉⁿ]))
sol["p_plus"] = canonicalize_2d(model[:p⁺])
sol["p_minus"] = canonicalize_2d(model[:p⁻])
sol["p_ch"] = canonicalize_2d(model[:pᶜʰ])
sol["p_dis"] = canonicalize_2d(model[:pᵈⁱˢ])
sol["e"] = canonicalize_2d(model[:e])
sol["d_shed"] = canonicalize_2d(model[:dˢʰᵉᵈ])
sol["xbar"] = value(model[:x̅])
sol["omega_plus"] = collect(value.(model[:ω⁺]))
if haskey(model, :f_p)
    sol["f_p"] = canonicalize_2d(model[:f_p])
end
if haskey(model, :f_q)
    sol["f_q"] = canonicalize_2d(model[:f_q])
end

# include data needed for observations
sol["PV"] = canonicalize_data_2d(data["PV"])
sol["D"] = canonicalize_data_2d(data["D"])
sol["spot"] = data["spot"]
sol["y_im"] = data["yⁱᵐ"]
sol["y_ex"] = data["yᵉˣ"]
sol["alpha_grid"] = data["αᵍʳⁱᵈ"]
sol["beta"] = b
sol["Delta"] = data["Delta"]

outdir = joinpath(root, "outputs")
isdir(outdir) || mkpath(outdir)
outfile = joinpath(outdir, "opt_solution.json")
open(outfile, "w") do io
    JSON.print(io, sol)
end

println("Saved solution to ", outfile)
