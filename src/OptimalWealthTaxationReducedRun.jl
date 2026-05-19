using JuMP
using Parameters
using PyPlot
const MOI = JuMP.MOI

PyPlot.ioff()

"""
    ReducedOptimalWealthTaxParams(; kwargs...)

Compatibility alias for the shared `NoWealthTaxation.ModelParams` container.

Optional parameters:
- See `NoWealthTaxation.ModelParams`; the reduced solver now reads all of its parameters from that single shared container.
"""
const ReducedOptimalWealthTaxParams = NoWealthTaxation.ModelParams

default_optimal_wealth_taxation_reduced_output_dir() = joinpath(normpath(joinpath(@__DIR__, "..")), "outputs", "optimal_wealth_taxation_reduced")

_positive_floor(x, floor) = ifelse(x > floor, x, zero(x) + floor)

function reduced_segment_integral(base, decay, dt::Real)
    if abs(decay) <= sqrt(eps(Float64))
        return base * dt
    end
    return base * (1.0 - exp(-decay * dt)) / decay
end

function reduced_time_grid(p::ReducedOptimalWealthTaxParams)
    p.N >= 2 || throw(ArgumentError("N must be at least 2"))
    p.T > 0.0 || throw(ArgumentError("T must be strictly positive"))
    return collect(range(0.0, p.T, length = p.N))
end

function reduced_production_terms(k, p::ReducedOptimalWealthTaxParams)
    k_eff = _positive_floor(k, p.min_positive)
    scale = p.A * p.n^p.η * p.l^(1.0 - p.θ - p.η)
    labor_scale = p.A * p.l^(1.0 - p.θ - p.η) * p.η * p.n^(p.η - 1.0)
    F = scale * k_eff^p.θ
    Fk = scale * p.θ * k_eff^(p.θ - 1.0)
    Fn = labor_scale * k_eff^p.θ
    return (; F, Fk, Fn)
end

function reduced_control_seed(p::ReducedOptimalWealthTaxParams)
    terms = reduced_production_terms(p.k0, p)
    upper = max(terms.Fk - p.δ, 0.0)
    if p.β < 1.0
        upper = min(upper, max((p.ρ - 100.0 * p.min_positive) / (1.0 - p.β), 0.0))
    end
    return 0.5 * upper
end

function reduced_with_horizon_mesh(p::ReducedOptimalWealthTaxParams, T::Real, N::Integer)
    kwargs = (; (name => getfield(p, name) for name in fieldnames(typeof(p)))...)
    return ReducedOptimalWealthTaxParams(; kwargs..., T = Float64(T), N = Int(N))
end

function reduced_interpolate_control_guess(old_t::AbstractVector{<:Real}, old_r::AbstractVector{<:Real}, new_t::AbstractVector{<:Real})
    isempty(old_t) && throw(ArgumentError("old_t must be non-empty"))
    length(old_t) == length(old_r) || throw(ArgumentError("old_t and old_r must have the same length"))

    guess = zeros(Float64, length(new_t))
    cursor = 1
    for (i, t) in enumerate(new_t)
        while cursor < length(old_t) - 1 && old_t[cursor + 1] < t
            cursor += 1
        end
        guess[i] = if t <= old_t[1]
            Float64(old_r[1])
        elseif t >= old_t[end]
            Float64(old_r[end])
        else
            t0 = Float64(old_t[cursor])
            t1 = Float64(old_t[cursor + 1])
            w = (Float64(t) - t0) / (t1 - t0)
            (1.0 - w) * Float64(old_r[cursor]) + w * Float64(old_r[cursor + 1])
        end
    end
    return guess
end

function default_reduced_T_stages(p::ReducedOptimalWealthTaxParams)
    p.T > 0.0 || throw(ArgumentError("T must be strictly positive"))
    initial_T = clamp(p.continuation_initial_T, p.min_positive, p.T)
    if p.T <= initial_T + sqrt(eps(Float64))
        return [Float64(p.T)]
    end

    stages = Float64[Float64(initial_T)]
    current = Float64(initial_T)
    while current < p.T
        next_T = min(p.T, 2.0 * current)
        if next_T <= stages[end] + sqrt(eps(Float64))
            break
        end
        push!(stages, next_T)
        current = next_T
    end
    return stages
end

function default_reduced_N_stages(p::ReducedOptimalWealthTaxParams)
    p.N >= 2 || throw(ArgumentError("N must be at least 2"))
    initial_N = clamp(p.continuation_initial_N, 2, p.N)
    if p.N <= initial_N
        return [Int(p.N)]
    end

    stages = Int[initial_N]
    current = stages[1]
    while current < p.N
        next_N = min(p.N, 2 * current - 1)
        if next_N <= stages[end]
            break
        end
        push!(stages, next_N)
        current = next_N
    end
    return stages
end

function reduced_stage_schedule(p::ReducedOptimalWealthTaxParams;
    continuation_T_stages::Union{Nothing, AbstractVector{<:Real}} = nothing,
    continuation_N_stages::Union{Nothing, AbstractVector{<:Integer}} = nothing)
    T_stages = continuation_T_stages === nothing ? default_reduced_T_stages(p) : Float64.(continuation_T_stages)
    N_stages = continuation_N_stages === nothing ? default_reduced_N_stages(p) : Int.(continuation_N_stages)
    isempty(T_stages) && throw(ArgumentError("continuation_T_stages must be non-empty"))
    isempty(N_stages) && throw(ArgumentError("continuation_N_stages must be non-empty"))

    schedule = Tuple{Float64, Int}[]
    base_N = first(N_stages)
    for T in T_stages
        push!(schedule, (Float64(T), base_N))
    end
    for N in N_stages[2:end]
        push!(schedule, (p.T, Int(N)))
    end

    if schedule[end] != (p.T, p.N)
        push!(schedule, (p.T, p.N))
    end
    return unique(schedule)
end

function reduced_follower_response(r::NTuple{N, T}, p::ReducedOptimalWealthTaxParams, t::AbstractVector{<:Real}) where {N, T <: Real}
    S = promote_type(T, Float64)
    R = zeros(S, N)
    c = zeros(S, N)
    denominator = zero(S)

    for i in 1:N-1
        dt = t[i + 1] - t[i]
        decay = (p.ρ + (p.β - 1.0) * r[i]) / p.β
        base = exp(-(p.ρ * t[i] + (p.β - 1.0) * R[i]) / p.β)
        denominator += reduced_segment_integral(base, decay, dt)
        R[i + 1] = R[i] + r[i] * dt
    end

    tail_decay = p.ρ + (p.β - 1.0) * r[end]
    tail_decay_eff = _positive_floor(tail_decay, p.min_positive)
    tail_base = exp(-(p.ρ * t[end] + (p.β - 1.0) * R[end]) / p.β)
    denominator += tail_base * p.β / tail_decay_eff

    a0 = p.k0 + p.n * p.q0
    c0 = a0 / denominator
    for i in 1:N
        c[i] = c0 * exp((R[i] - p.ρ * t[i]) / p.β)
    end

    return (; c, c0, R, tail_decay)
end

function reduced_backward_euler_state_step(k_now, q_now, r_next, c_next, dt, p::ReducedOptimalWealthTaxParams)
    k_next = k_now
    q_next = q_now

    for _ in 1:p.reduced_implicit_iterations
        terms_next = reduced_production_terms(k_next, p)
        drift_next = terms_next.Fk - p.δ
        q_denom = 1.0 - dt * drift_next
        q_denom_eff = ifelse(abs(q_denom) > p.min_positive, q_denom, q_denom + p.min_positive)
        q_next = (q_now - dt * terms_next.Fn) / q_denom_eff

        k_denom = 1.0 - dt * r_next
        k_denom_eff = ifelse(abs(k_denom) > p.min_positive, k_denom, k_denom + p.min_positive)
        k_next = (k_now + dt * (q_next * (r_next - drift_next) + terms_next.Fn - c_next)) / k_denom_eff
    end

    return k_next, q_next
end

function reduced_simulation(r::NTuple{N, T}, p::ReducedOptimalWealthTaxParams) where {N, T <: Real}
    N == p.N || throw(ArgumentError("control tuple length must match p.N"))
    S = promote_type(T, Float64)
    t = reduced_time_grid(p)
    c_path = reduced_follower_response(r, p, t)
    k = zeros(S, N)
    q = zeros(S, N)
    a = zeros(S, N)
    x = zeros(S, N)
    upper_gap = zeros(S, N)
    Fk = zeros(S, N)
    objective = zero(S)

    k[1] = zero(S) + p.k0
    q[1] = zero(S) + p.q0
    discount = exp.(-p.ρ .* t[1:end-1])

    for i in 1:N
        terms = reduced_production_terms(k[i], p)
        Fk[i] = terms.Fk
        a[i] = k[i] + p.n * q[i]
        x[i] = terms.F - p.δ * k[i] - r[i] * k[i] - q[i] * (r[i] - (terms.Fk - p.δ)) - terms.Fn
        upper_gap[i] = max(terms.Fk - p.δ, 0.0) - r[i]

        if i < N
            dt = t[i + 1] - t[i]
            k[i + 1], q[i + 1] = reduced_backward_euler_state_step(k[i], q[i], r[i + 1], c_path.c[i + 1], dt, p)
            x_eff = _positive_floor(x[i], p.min_positive)
            c_eff = _positive_floor(c_path.c[i], p.min_positive)
            utility = abs(p.β - 1.0) <= 1.0e-12 ? log(c_eff) : c_eff^(1.0 - p.β) / (1.0 - p.β)
            objective += dt * discount[i] * (p.γ * log(x_eff) + utility)
        end
    end

    return (; t, k, q, a, c = c_path.c, x, upper_gap, Fk, tail_decay = c_path.tail_decay, objective)
end

function reduced_control_path(control_path::Real, p::ReducedOptimalWealthTaxParams)
    return fill(Float64(control_path), p.N)
end

function reduced_control_path(control_path::AbstractVector{<:Real}, p::ReducedOptimalWealthTaxParams)
    length(control_path) == p.N || throw(ArgumentError("control_path must have length p.N"))
    return Float64.(control_path)
end

function reduced_result_from_control_path(control_path, p::ReducedOptimalWealthTaxParams;
    status = :NOT_OPTIMIZED,
    success::Union{Nothing, Bool} = nothing,
    objective_override = nothing)
    r_tilde = reduced_control_path(control_path, p)
    sim = reduced_simulation(Tuple(r_tilde), p)
    tau_k = similar(r_tilde)
    for i in eachindex(r_tilde)
        denom = sim.Fk[i] - p.δ
        tau_k[i] = denom <= p.min_positive ? 1.0 : clamp(1.0 - r_tilde[i] / denom, 0.0, 1.0)
    end

    feasible = minimum(sim.x) > 0.0 &&
        minimum(sim.k) >= -1.0e-7 &&
        minimum(sim.q) >= -1.0e-7 &&
        minimum(sim.upper_gap) >= -1.0e-7 &&
        sim.tail_decay > 0.0
    objective = objective_override === nothing ? sim.objective : objective_override
    success_flag = success === nothing ? feasible : success

    return (; success = success_flag, feasible, status, objective, t = sim.t, r_tilde, tau_k,
        k = collect(sim.k), q = collect(sim.q), a = collect(sim.a), c = collect(sim.c), x = collect(sim.x),
        control_gap = collect(sim.upper_gap), tail_decay = sim.tail_decay)
end

reduced_objective(r::NTuple, p::ReducedOptimalWealthTaxParams) = reduced_simulation(r, p).objective
reduced_x_constraint(node::Int, r::NTuple, p::ReducedOptimalWealthTaxParams) = reduced_simulation(r, p).x[node]
reduced_upper_gap_constraint(node::Int, r::NTuple, p::ReducedOptimalWealthTaxParams) = reduced_simulation(r, p).upper_gap[node]
reduced_k_constraint(node::Int, r::NTuple, p::ReducedOptimalWealthTaxParams) = reduced_simulation(r, p).k[node]
reduced_q_constraint(node::Int, r::NTuple, p::ReducedOptimalWealthTaxParams) = reduced_simulation(r, p).q[node]
reduced_tail_decay_constraint(r::NTuple, p::ReducedOptimalWealthTaxParams) = reduced_simulation(r, p).tail_decay

function reduced_operator_call(name::Symbol, r)
    return Expr(:call, name, r...)
end

function build_reduced_optimal_wealth_taxation_model(p::ReducedOptimalWealthTaxParams; initial_r::Union{Nothing, AbstractVector{<:Real}} = nothing)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", p.max_iter)
    set_optimizer_attribute(model, "print_level", p.ipopt_print_level)

    r_seed = reduced_control_seed(p)
    @variable(model, r[1:p.N] >= 0.0, start = r_seed)
    if initial_r !== nothing
        length(initial_r) == p.N || throw(ArgumentError("initial_r must have length p.N"))
        for i in 1:p.N
            set_start_value(r[i], Float64(initial_r[i]))
        end
    end

    objective_name = :reduced_optimal_wealth_taxation_objective
    JuMP.register(model, objective_name, p.N, (args...) -> reduced_objective(args, p); autodiff = true)
    set_nonlinear_objective(model, MOI.MAX_SENSE, reduced_operator_call(objective_name, r))

    for node in 1:p.N
        x_name = Symbol("reduced_optimal_wealth_taxation_x_", node)
        bound_name = Symbol("reduced_optimal_wealth_taxation_bound_", node)
        k_name = Symbol("reduced_optimal_wealth_taxation_k_", node)
        q_name = Symbol("reduced_optimal_wealth_taxation_q_", node)
        JuMP.register(model, x_name, p.N, (args...) -> reduced_x_constraint(node, args, p); autodiff = true)
        JuMP.register(model, bound_name, p.N, (args...) -> reduced_upper_gap_constraint(node, args, p); autodiff = true)
        JuMP.register(model, k_name, p.N, (args...) -> reduced_k_constraint(node, args, p); autodiff = true)
        JuMP.register(model, q_name, p.N, (args...) -> reduced_q_constraint(node, args, p); autodiff = true)
        add_nonlinear_constraint(model, Expr(:call, :>=, reduced_operator_call(x_name, r), p.min_positive))
        add_nonlinear_constraint(model, Expr(:call, :>=, reduced_operator_call(bound_name, r), 0.0))
        add_nonlinear_constraint(model, Expr(:call, :>=, reduced_operator_call(k_name, r), 0.0))
        add_nonlinear_constraint(model, Expr(:call, :>=, reduced_operator_call(q_name, r), 0.0))
    end

    tail_name = :reduced_optimal_wealth_taxation_tail_decay
    JuMP.register(model, tail_name, p.N, (args...) -> reduced_tail_decay_constraint(args, p); autodiff = true)
    add_nonlinear_constraint(model, Expr(:call, :>=, reduced_operator_call(tail_name, r), p.min_positive))

    return model
end

function reduced_result_is_acceptable(result)
    return result.status in (MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ITERATION_LIMIT) &&
    isfinite(result.objective) &&
    result.feasible
end

function solve_reduced_optimal_wealth_taxation_stage(p::ReducedOptimalWealthTaxParams; initial_r::Union{Nothing, AbstractVector{<:Real}} = nothing)
    model = build_reduced_optimal_wealth_taxation_model(p; initial_r = initial_r)
    optimize!(model)
    return extract_reduced_optimal_wealth_taxation_result(model, p)
end

function extract_reduced_optimal_wealth_taxation_result(model::Model, p::ReducedOptimalWealthTaxParams)
    status = termination_status(model)
    evaluated = reduced_result_from_control_path(value.(model[:r]), p;
        status = status,
        success = status in (MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED),
        objective_override = objective_value(model))
    return (; evaluated..., success = evaluated.success && evaluated.feasible)
end

function writeReducedOptimalWealthTaxationResultCSV(result, file_path::AbstractString)
    headers = ["t", "k", "q", "a", "c", "x", "r_tilde", "tau_k", "control_gap"]
    rows = ((result.t[i], result.k[i], result.q[i], result.a[i], result.c[i], result.x[i], result.r_tilde[i], result.tau_k[i], result.control_gap[i]) for i in eachindex(result.t))
    return write_csv_table(file_path, headers, rows)
end

"""
    plotReducedOptimalWealthTaxationSolution(result, title, filename; force=false, half=false)

Create the main reduced-solver plot with six stacked panels in the same visual style used by the legacy plots.
Only saves to PNG and does not display the figure.
"""
function plotReducedOptimalWealthTaxationSolution(result, title, filename; force::Bool = false, half::Bool = false)
    if !result.success && !force
        println("Cannot visualize - solution failed to converge (set force=true to override)")
        return nothing
    end

    inds = if half
        T_max = maximum(result.t)
        result.t .<= T_max / 2
    else
        trues(length(result.t))
    end

    t_plot = result.t[inds]
    k_plot = result.k[inds]
    q_plot = result.q[inds]
    a_plot = result.a[inds]
    c_plot = result.c[inds]
    r_tilde_plot = result.r_tilde[inds]
    tau_k_plot = result.tau_k[inds]

    fig, ax = PyPlot.subplots(6, 1, figsize = (12, 14))
    fig.suptitle(title, fontsize = 18, y = 0.98)

    ax[1].plot(t_plot, k_plot, "b-", linewidth = 2)
    ax[1].set_title("Capital k Trajectory")
    ax[1].set_xlabel("Time")
    ax[1].grid(true)

    ax[2].plot(t_plot, q_plot, color = "orange", linewidth = 2)
    ax[2].set_title("State q Trajectory")
    ax[2].set_xlabel("Time")
    ax[2].grid(true)

    ax[3].plot(t_plot, a_plot, "k-", linewidth = 2)
    ax[3].set_title("Assets a Trajectory")
    ax[3].set_xlabel("Time")
    ax[3].grid(true)

    ax[4].plot(t_plot, c_plot, "m-", linewidth = 2)
    ax[4].set_title("Consumption c Trajectory")
    ax[4].set_xlabel("Time")
    ax[4].grid(true)

    ax[5].plot(t_plot, r_tilde_plot, "g-", linewidth = 2)
    ax[5].set_title("Effective Interest Rate r̃")
    ax[5].set_xlabel("Time")
    ax[5].grid(true)

    ax[6].plot(t_plot, tau_k_plot, "c-", linewidth = 2)
    ax[6].set_title("Capital Tax Rate τ")
    ax[6].set_xlabel("Time")
    ax[6].grid(true)

    PyPlot.tight_layout(rect = (0, 0, 1, 0.97))
    PyPlot.savefig(filename, dpi = 300, bbox_inches = "tight")
    PyPlot.close(fig)
    println("✓ Plot saved as '$(filename)'")
    return filename
end

"""
    _solveReducedOptimalWealthTaxation(; output_dir=default_optimal_wealth_taxation_reduced_output_dir(), progress=true, model_kwargs=(;), use_continuation=nothing, continuation_T_stages=nothing, continuation_N_stages=nothing, continuation_max_T_step=nothing, continuation_min_T_step=nothing)

Runs the reduced-form `OptimalWealthTaxation` strategy: optimize only over the discretized path of `r_tilde`, recover the follower best response analytically, and integrate the leader states forward.
"""
function _solveReducedOptimalWealthTaxation(; output_dir::AbstractString = default_optimal_wealth_taxation_reduced_output_dir(), progress::Bool = true, model_kwargs::NamedTuple = (;), use_continuation::Union{Nothing, Bool} = nothing, continuation_T_stages::Union{Nothing, AbstractVector{<:Real}} = nothing, continuation_N_stages::Union{Nothing, AbstractVector{<:Integer}} = nothing, continuation_max_T_step::Union{Nothing, Real} = nothing, continuation_min_T_step::Union{Nothing, Real} = nothing)
    mkpath(output_dir)
    effective_model_kwargs = model_kwargs
    p = NoWealthTaxation.ModelParams(; effective_model_kwargs...)
    effective_use_continuation = use_continuation === nothing ? p.use_continuation : use_continuation
    effective_continuation_max_T_step = continuation_max_T_step === nothing ? p.continuation_max_T_step : Float64(continuation_max_T_step)
    effective_continuation_min_T_step = continuation_min_T_step === nothing ? p.continuation_min_T_step : Float64(continuation_min_T_step)

    progress && println("OptimalWealthTaxation reduced run start")
    progress && println("  output_dir=$(output_dir)")
    progress && println("  model_kwargs=$(effective_model_kwargs)")

    stage_rows = Tuple{Float64, Int, String, Bool, Float64, Float64, Float64, Float64}[]
    if effective_use_continuation
        schedule = reduced_stage_schedule(p;
            continuation_T_stages = continuation_T_stages,
            continuation_N_stages = continuation_N_stages)
        progress && println("  continuation_schedule=$(schedule)")

        current_guess = nothing
        current_t = nothing
        current_T = 0.0
        current_N = 0
        result = nothing
        for (stage_idx, (stage_T, stage_N)) in enumerate(schedule)
            progress && println("  stage $(stage_idx)/$(length(schedule)) -> target T=$(stage_T) N=$(stage_N)")
            if current_guess !== nothing && stage_N == current_N && stage_T > current_T + 1.0e-12
                attempted_step = min(effective_continuation_max_T_step, stage_T - current_T)
                while current_T < stage_T - 1.0e-12
                    trial_T = min(stage_T, current_T + attempted_step)
                    stage_p = reduced_with_horizon_mesh(p, trial_T, stage_N)
                    stage_t = reduced_time_grid(stage_p)
                    stage_guess = reduced_interpolate_control_guess(current_t, current_guess, stage_t)
                    stage_result = solve_reduced_optimal_wealth_taxation_stage(stage_p; initial_r = stage_guess)
                    push!(stage_rows, (trial_T, stage_N, string(stage_result.status), stage_result.success, stage_result.objective, minimum(stage_result.x), minimum(stage_result.control_gap), stage_result.tail_decay))
                    progress && println("    trial T=$(trial_T) step=$(attempted_step) status=$(stage_result.status) objective=$(stage_result.objective) min_x=$(minimum(stage_result.x)) min_gap=$(minimum(stage_result.control_gap))")

                    if reduced_result_is_acceptable(stage_result)
                        current_guess = stage_result.r_tilde
                        current_t = stage_result.t
                        current_T = trial_T
                        current_N = stage_N
                        result = stage_result
                        attempted_step = min(effective_continuation_max_T_step, stage_T - current_T)
                    else
                        attempted_step *= 0.5
                        if attempted_step < effective_continuation_min_T_step - 1.0e-12
                            error("Reduced continuation failed near target T=$(stage_T), N=$(stage_N); smallest attempted horizon step=$(attempted_step * 2) produced status=$(stage_result.status)")
                        end
                    end
                end
                continue
            end

            stage_p = reduced_with_horizon_mesh(p, stage_T, stage_N)
            stage_t = reduced_time_grid(stage_p)
            stage_guess = current_guess === nothing ? nothing : reduced_interpolate_control_guess(current_t, current_guess, stage_t)
            stage_result = solve_reduced_optimal_wealth_taxation_stage(stage_p; initial_r = stage_guess)
            push!(stage_rows, (stage_T, stage_N, string(stage_result.status), stage_result.success, stage_result.objective, minimum(stage_result.x), minimum(stage_result.control_gap), stage_result.tail_decay))
            progress && println("    status=$(stage_result.status) objective=$(stage_result.objective) min_x=$(minimum(stage_result.x)) min_gap=$(minimum(stage_result.control_gap))")

            if !reduced_result_is_acceptable(stage_result)
                error("Reduced continuation failed at stage T=$(stage_T), N=$(stage_N) with status=$(stage_result.status)")
            end

            current_guess = stage_result.r_tilde
            current_t = stage_result.t
            current_T = stage_T
            current_N = stage_N
            result = stage_result
        end
    else
        result = solve_reduced_optimal_wealth_taxation_stage(p)
        push!(stage_rows, (p.T, p.N, string(result.status), result.success, result.objective, minimum(result.x), minimum(result.control_gap), result.tail_decay))
    end

    csv_path = joinpath(output_dir, "OptimalWealthTaxationReduced_solution.csv")
    plot_path = joinpath(output_dir, "OptimalWealthTaxationReduced_solution.png")
    summary_path = joinpath(output_dir, "OptimalWealthTaxationReduced_summary.csv")
    stages_path = joinpath(output_dir, "OptimalWealthTaxationReduced_stages.csv")
    writeReducedOptimalWealthTaxationResultCSV(result, csv_path)
    plotReducedOptimalWealthTaxationSolution(result, "OptimalWealthTaxation reduced path", plot_path; force = true, half = false)
    write_csv_table(summary_path,
        ["success", "status", "objective", "horizon_T", "mesh_N", "use_continuation", "final_k", "final_q", "final_c", "min_x", "min_control_gap", "tail_decay", "solution_csv", "plot_png", "stages_csv"],
        [(result.success, string(result.status), result.objective, p.T, p.N, effective_use_continuation, result.k[end], result.q[end], result.c[end], minimum(result.x), minimum(result.control_gap), result.tail_decay, csv_path, plot_path, stages_path)])
    write_csv_table(stages_path,
        ["stage_T", "stage_N", "status", "success", "objective", "min_x", "min_control_gap", "tail_decay"],
        stage_rows)

    if progress
        println("OptimalWealthTaxation reduced run complete")
        println("  success=$(result.success) status=$(result.status) objective=$(result.objective)")
        println("  final  k=$(result.k[end]) q=$(result.q[end]) c=$(result.c[end])")
        println("  mins   x=$(minimum(result.x)) control_gap=$(minimum(result.control_gap)) tail_decay=$(result.tail_decay)")
        println("  solution_csv=$(csv_path)")
        println("  plot_png=$(plot_path)")
        println("  summary_csv=$(summary_path)")
        println("  stages_csv=$(stages_path)")
    end

    return (; params = p, result = result, solution_csv = csv_path, plot_png = plot_path, summary_csv = summary_path, stages_csv = stages_path, stage_rows)
end

"""
    _checkReducedOptimalWealthTaxationFeasibility(control_path; output_dir=default_optimal_wealth_taxation_reduced_output_dir(), progress=true, model_kwargs=(;), filename_prefix="OptimalWealthTaxationReduced_feasibility")

Evaluate a candidate `r_tilde` path without optimization and report whether the induced reduced-model trajectory satisfies the state and control constraints.
"""
function _checkReducedOptimalWealthTaxationFeasibility(control_path;
    output_dir::AbstractString = default_optimal_wealth_taxation_reduced_output_dir(),
    progress::Bool = true,
    model_kwargs::NamedTuple = (;),
    filename_prefix::AbstractString = "OptimalWealthTaxationReduced_feasibility")
    mkpath(output_dir)
    p = NoWealthTaxation.ModelParams(; model_kwargs...)
    result = reduced_result_from_control_path(control_path, p)

    csv_path = joinpath(output_dir, "$(filename_prefix)_solution.csv")
    plot_path = joinpath(output_dir, "$(filename_prefix)_solution.png")
    summary_path = joinpath(output_dir, "$(filename_prefix)_summary.csv")
    writeReducedOptimalWealthTaxationResultCSV(result, csv_path)
    plotReducedOptimalWealthTaxationSolution(result, "OptimalWealthTaxation reduced feasibility path", plot_path; force = true, half = false)
    write_csv_table(summary_path,
        ["feasible", "objective", "horizon_T", "mesh_N", "final_k", "final_q", "final_c", "min_x", "min_k", "min_q", "min_control_gap", "tail_decay", "solution_csv", "plot_png"],
        [(result.feasible, result.objective, p.T, p.N, result.k[end], result.q[end], result.c[end], minimum(result.x), minimum(result.k), minimum(result.q), minimum(result.control_gap), result.tail_decay, csv_path, plot_path)])

    if progress
        println("OptimalWealthTaxation reduced feasibility check complete")
        println("  feasible=$(result.feasible) objective=$(result.objective)")
        println("  final  k=$(result.k[end]) q=$(result.q[end]) c=$(result.c[end])")
        println("  mins   x=$(minimum(result.x)) k=$(minimum(result.k)) q=$(minimum(result.q)) control_gap=$(minimum(result.control_gap)) tail_decay=$(result.tail_decay)")
        println("  solution_csv=$(csv_path)")
        println("  plot_png=$(plot_path)")
        println("  summary_csv=$(summary_path)")
    end

    return (; params = p, result = result, solution_csv = csv_path, plot_png = plot_path, summary_csv = summary_path)
end