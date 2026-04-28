import Pkg

root = dirname(@__FILE__)
proj_path = joinpath(root, "DP_CC_OPF")
if isdir(proj_path) && isfile(joinpath(proj_path, "Project.toml"))
    Pkg.activate(proj_path)
else
    Pkg.activate(root)
end

try
    Pkg.instantiate()
catch err
    @warn "Pkg.instantiate failed" err
end

using JSON, Gurobi, JuMP

payload_path = ARGS[1]
output_path = ARGS[2]

payload = JSON.parsefile(payload_path)

prices = Float64.(payload["prices"])
D = Float64.(payload["D"])
PV = Float64.(payload["PV"])
soc = Float64.(payload["soc"])
eta = Float64(payload["eta"])
E_max = Float64.(payload["E_max"])
p_ch_max = Float64.(payload["p_ch_max"])
p_dis_max = Float64.(payload["p_dis_max"])
y_im = Float64(payload["y_im"])
y_ex = Float64(payload["y_ex"])

const GRB_ENV = Gurobi.Env(Dict{String,Any}("OutputFlag" => 0, "MIPGap" => 1e-4))

function solve_consumer(price, demand, pv, soc0, emax, pchmax, pdismax, eta, y_im, y_ex)
    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_silent(model)

    @variable(model, 0 <= e <= emax)
    @variable(model, 0 <= p_ch <= pchmax)
    @variable(model, 0 <= p_dis <= pdismax)
    @variable(model, 0 <= p_plus)
    @variable(model, 0 <= p_minus)
    @variable(model, 0 <= q_plus)
    @variable(model, 0 <= q_minus)
    @variable(model, 0 <= d_shed)

    @objective(model, Min, (price + y_im) * p_plus - (price - y_ex) * p_minus + 1.25 * 75 * d_shed)
    @constraint(model, power_balance, p_plus - p_minus + pv - demand + p_dis - p_ch + d_shed == 0)
    @constraint(model, battery_balance, e == soc0 + eta * p_ch - p_dis / eta)
    @constraint(model, reactive_plus, q_plus == 0.5 * p_plus)
    @constraint(model, reactive_minus, q_minus == 0.5 * p_minus)

    optimize!(model)

    if !has_values(model)
        return Dict(
            "p_plus" => 0.0,
            "p_minus" => 0.0,
            "p_ch" => 0.0,
            "p_dis" => 0.0,
            "e" => soc0,
            "d_shed" => demand - pv,
            "f_p" => 0.0,
            "f_q" => 0.0,
            "next_soc" => soc0,
        )
    end

    p_plus_v = value(p_plus)
    p_minus_v = value(p_minus)
    p_ch_v = value(p_ch)
    p_dis_v = value(p_dis)
    e_v = value(e)
    d_shed_v = value(d_shed)

    return Dict(
        "p_plus" => p_plus_v,
        "p_minus" => p_minus_v,
        "p_ch" => p_ch_v,
        "p_dis" => p_dis_v,
        "e" => e_v,
        "d_shed" => d_shed_v,
        "f_p" => p_plus_v - p_minus_v,
        "f_q" => 0.5 * (p_plus_v - p_minus_v),
        "next_soc" => e_v,
    )
end

results = Dict{String,Any}()
results["p_plus"] = Float64[]
results["p_minus"] = Float64[]
results["p_ch"] = Float64[]
results["p_dis"] = Float64[]
results["e"] = Float64[]
results["d_shed"] = Float64[]
results["f_p"] = Float64[]
results["f_q"] = Float64[]
results["next_soc"] = Float64[]

for i in eachindex(prices)
    out = solve_consumer(prices[i], D[i], PV[i], soc[i], E_max[i], p_ch_max[i], p_dis_max[i], eta, y_im, y_ex)
    push!(results["p_plus"], out["p_plus"])
    push!(results["p_minus"], out["p_minus"])
    push!(results["p_ch"], out["p_ch"])
    push!(results["p_dis"], out["p_dis"])
    push!(results["e"], out["e"])
    push!(results["d_shed"], out["d_shed"])
    push!(results["f_p"], out["f_p"])
    push!(results["f_q"], out["f_q"])
    push!(results["next_soc"], out["next_soc"])
end

open(output_path, "w") do io
    JSON.print(io, results)
end