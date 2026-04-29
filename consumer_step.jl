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

function to_matrix(x, rows::Int, cols::Int, name::String)
    if x isa AbstractVector && !isempty(x) && x[1] isa AbstractVector
        mat = reduce(hcat, [Float64.(row) for row in x])'
    else
        flat = Float64.(x)
        if length(flat) != rows * cols
            error("Unexpected length for $name: $(length(flat)) != $(rows * cols)")
        end
        mat = reshape(flat, rows, cols)
    end

    if size(mat) == (rows, cols)
        return mat
    elseif size(mat) == (cols, rows)
        return permutedims(mat)
    end

    error("Unexpected shape for $name: $(size(mat))")
end

function to_vector(x, name::String)
    return Float64.(x)
end

soc = to_vector(payload["soc"], "soc")
eta = Float64(payload["eta"])
E_max = to_vector(payload["E_max"], "E_max")
p_ch_max = to_vector(payload["p_ch_max"], "p_ch_max")
p_dis_max = to_vector(payload["p_dis_max"], "p_dis_max")
y_im = to_vector(payload["y_im"], "y_im")
y_ex = to_vector(payload["y_ex"], "y_ex")
prices = to_matrix(payload["prices"], length(soc), length(y_im), "prices")
D = to_matrix(payload["D"], length(soc), length(y_im), "D")
PV = to_matrix(payload["PV"], length(soc), length(y_im), "PV")

const GRB_ENV = Gurobi.Env(Dict{String,Any}("OutputFlag" => 0, "MIPGap" => 1e-4))

function solve_consumer_day(price_row, demand_row, pv_row, soc0, emax, pchmax, pdismax, eta, y_im, y_ex)
    T = length(price_row)
    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_silent(model)

    # DC-side battery formulation matching KKT conditions in model.jl:
    # - Power balance: p+ - p- + PV - D + d_shed + eta*p_dis - p_ch/eta == 0
    # - Battery balance: e[t] = e[t-1] + p_ch[t] - p_dis[t]  (no eta on DC side)
    # - Cyclic: e[1] = e[T] + p_ch[1] - p_dis[1]
    # - Bounds: p_ch <= emax/2, p_dis <= emax/2  (DC-side bounds)
    p_ch_dc_max = emax / 2.0
    p_dis_dc_max = emax / 2.0

    @variable(model, 0 <= e[1:T] <= emax)
    @variable(model, 0 <= p_ch[1:T] <= p_ch_dc_max)
    @variable(model, 0 <= p_dis[1:T] <= p_dis_dc_max)
    @variable(model, 0 <= p_plus[1:T])
    @variable(model, 0 <= p_minus[1:T])
    @variable(model, 0 <= q_plus[1:T])
    @variable(model, 0 <= q_minus[1:T])
    @variable(model, 0 <= d_shed[1:T])

    # Paper Eq: min Σ_t [ x_it·(p⁺ - p⁻) + α_shed·d_shed ]
    # Tariffs are in the upper-level cost balance, not the lower level.
    @objective(model, Min, sum(price_row[t] * (p_plus[t] - p_minus[t]) + 1.25 * 75 * d_shed[t] for t in 1:T))

    # Power balance with DC-side battery (eta in power terms)
    @constraint(model, power_balance[t=1:T], p_plus[t] - p_minus[t] + pv_row[t] - demand_row[t] + d_shed[t] + eta * p_dis[t] - p_ch[t] / eta == 0)

    # Cyclic battery constraint: e[1] = e[T] + p_ch[1] - p_dis[1]
    @constraint(model, battery_balance_1, e[1] == e[T] + p_ch[1] - p_dis[1])
    if T > 1
        @constraint(model, battery_balance[t=2:T], e[t] == e[t - 1] + p_ch[t] - p_dis[t])
    end
    @constraint(model, reactive_plus[t=1:T], q_plus[t] == 0.5 * p_plus[t])
    @constraint(model, reactive_minus[t=1:T], q_minus[t] == 0.5 * p_minus[t])

    optimize!(model)

    if !has_values(model)
        return Dict(
            "p_im" => zeros(T),
            "p_ex" => zeros(T),
            "p_pen" => zeros(T),
            "p_plus" => zeros(T),
            "p_minus" => zeros(T),
            "p_ch" => zeros(T),
            "p_dis" => zeros(T),
            "e" => fill(soc0, T),
            "d_shed" => max.(0.0, demand_row .- pv_row),
            "f_p" => zeros(T),
            "f_q" => zeros(T),
            "next_soc" => soc0,
        )
    end

    p_plus_v = value.(p_plus)
    p_minus_v = value.(p_minus)
    p_ch_v = value.(p_ch)
    p_dis_v = value.(p_dis)
    e_v = value.(e)
    d_shed_v = value.(d_shed)

    return Dict(
        "p_im" => collect(p_plus_v),
        "p_ex" => collect(p_minus_v),
        "p_pen" => zeros(T),
        "p_plus" => collect(p_plus_v),
        "p_minus" => collect(p_minus_v),
        "p_ch" => collect(p_ch_v),
        "p_dis" => collect(p_dis_v),
        "e" => collect(e_v),
        "d_shed" => collect(d_shed_v),
        "f_p" => collect(p_plus_v .- p_minus_v),
        "f_q" => collect(0.5 .* (p_plus_v .- p_minus_v)),
        "next_soc" => e_v[end],
    )
end

results = Dict{String,Any}()
results["p_im"] = Vector{Vector{Float64}}()
results["p_ex"] = Vector{Vector{Float64}}()
results["p_pen"] = Vector{Vector{Float64}}()
results["p_plus"] = Vector{Vector{Float64}}()
results["p_minus"] = Vector{Vector{Float64}}()
results["p_ch"] = Vector{Vector{Float64}}()
results["p_dis"] = Vector{Vector{Float64}}()
results["e"] = Vector{Vector{Float64}}()
results["d_shed"] = Vector{Vector{Float64}}()
results["f_p"] = Vector{Vector{Float64}}()
results["f_q"] = Vector{Vector{Float64}}()
results["next_soc"] = Float64[]

for i in 1:length(soc)
    out = solve_consumer_day(prices[i, :], D[i, :], PV[i, :], soc[i], E_max[i], p_ch_max[i], p_dis_max[i], eta, y_im, y_ex)
    push!(results["p_im"], out["p_im"])
    push!(results["p_ex"], out["p_ex"])
    push!(results["p_pen"], out["p_pen"])
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