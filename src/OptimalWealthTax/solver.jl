struct CollocationResult
    success::Bool
    t::Vector{Float64}
    k::Vector{Float64}
    c::Vector{Float64}
    q::Vector{Float64}
    Λ1::Vector{Float64}
    Λ2::Vector{Float64}
    Λ3::Vector{Float64}
    r_tilde::Vector{Float64}
    x::Vector{Float64}
    steady::SteadyStateResult
    residual_norm::Float64
end

function transversality_metrics(result::CollocationResult, p::ModelParams)
    discount = exp.(-p.ρ .* result.t)
    k_tvc = discount .* result.Λ1 .* result.k
    c_tvc = discount .* result.Λ2 .* result.c
    q_tvc = discount .* result.Λ3 .* result.q
    terminal = (; k = k_tvc[end], c = c_tvc[end], q = q_tvc[end])
    return (; k = k_tvc, c = c_tvc, q = q_tvc, terminal)
end

function pack_solution(result::CollocationResult)
    N = length(result.t)
    z = zeros(6 * N)
    for i in 1:N
        offset = node_offset(i)
        z[offset + 1] = result.k[i]
        z[offset + 2] = result.c[i]
        z[offset + 3] = result.q[i]
        z[offset + 4] = result.Λ1[i]
        z[offset + 5] = result.Λ2[i]
        z[offset + 6] = result.Λ3[i]
    end
    return z
end

function with_initial_conditions(p::ModelParams, k0::Real, q0::Real)
    return ModelParams(A = p.A, θ = p.θ, η = p.η, β = p.β, ρ = p.ρ, δ = p.δ, γ = p.γ,
        n = p.n, l = p.l, k0 = Float64(k0), q0 = Float64(q0), T = p.T, N = p.N,
    max_iter = p.max_iter, residual_tolerance = p.residual_tolerance, mesh_power = p.mesh_power,
    min_positive = p.min_positive)
end

function with_horizon(p::ModelParams, T::Real, N::Integer)
    return ModelParams(A = p.A, θ = p.θ, η = p.η, β = p.β, ρ = p.ρ, δ = p.δ, γ = p.γ,
        n = p.n, l = p.l, k0 = p.k0, q0 = p.q0, T = Float64(T), N = Int(N),
        max_iter = p.max_iter, residual_tolerance = p.residual_tolerance, mesh_power = p.mesh_power,
        min_positive = p.min_positive)
end

node_offset(i::Int) = 6 * (i - 1)

function node_slice(z::AbstractVector, i::Int)
    offset = node_offset(i)
    return @view z[offset + 1:offset + 6]
end

function collocation_guess(p::ModelParams, steady::SteadyStateResult, tgrid::AbstractVector)
    N = length(tgrid)
    guess = zeros(6 * N)
    for (i, t) in enumerate(tgrid)
        w = tgrid[end] <= 0 ? 0.0 : t / tgrid[end]
        offset = node_offset(i)
        guess[offset + 1] = (1.0 - w) * p.k0 + w * steady.k
        guess[offset + 2] = (1.0 - w) * steady.c + w * steady.c
        guess[offset + 3] = (1.0 - w) * p.q0 + w * steady.q
        guess[offset + 4] = steady.Λ1
        guess[offset + 5] = steady.Λ2
        guess[offset + 6] = steady.Λ3
    end
    return guess
end

function interpolate_guess(old_t::AbstractVector, old_z::AbstractVector, new_t::AbstractVector)
    old_n = length(old_t)
    new_n = length(new_t)
    new_z = zeros(6 * new_n)
    for var in 1:6
        old_values = [old_z[node_offset(i) + var] for i in 1:old_n]
        cursor = 1
        for (j, t) in enumerate(new_t)
            while cursor < old_n - 1 && old_t[cursor + 1] < t
                cursor += 1
            end
            value = if t <= old_t[1]
                old_values[1]
            elseif t >= old_t[end]
                old_values[end]
            else
                t0 = old_t[cursor]
                t1 = old_t[cursor + 1]
                w = (t - t0) / (t1 - t0)
                (1.0 - w) * old_values[cursor] + w * old_values[cursor + 1]
            end
            new_z[node_offset(j) + var] = value
        end
    end
    return new_z
end

function interpolate_state(old_t::AbstractVector, old_z::AbstractVector, t::Real)
    tmp = interpolate_guess(old_t, old_z, [Float64(t)])
    return collect(node_slice(tmp, 1))
end

function rescale_time_grid(old_t::AbstractVector, new_T::Real)
    old_T = old_t[end]
    if old_T <= 0
        return zeros(length(old_t))
    end
    return collect(Float64(new_T) .* (old_t ./ old_T))
end

function collocation_grid(p::ModelParams, N::Int)
    ξ = range(0.0, 1.0, length = N)
    return collect(p.T .* (ξ .^ p.mesh_power))
end

function collocation_residual!(residual, z, p::ModelParams, steady::SteadyStateResult, tgrid::AbstractVector)
    N = length(tgrid)
    fill!(residual, 0.0)
    idx = 1
    scales = (
        max(abs(p.k0), abs(steady.k), 1.0),
        max(abs(steady.c), 1.0),
        max(abs(p.q0), abs(steady.q), 1.0),
        max(abs(steady.Λ1), 1.0),
        max(abs(steady.Λ2), 1.0),
        max(abs(steady.Λ3), 1.0),
    )

    y0 = node_slice(z, 1)
    residual[idx] = (y0[1] - p.k0) / scales[1]
    idx += 1
    residual[idx] = (y0[3] - p.q0) / scales[3]
    idx += 1
    for i in 1:N-1
        yi = collect(node_slice(z, i))
        yj = collect(node_slice(z, i + 1))
        fi = dynamics(yi, p)
        fj = dynamics(yj, p)
        h = tgrid[i + 1] - tgrid[i]
        if !(all(isfinite, fi) && all(isfinite, fj))
            residual[idx:idx + 5] .= 1e6
            idx += 6
            continue
        end
        for j in 1:6
            residual[idx] = (yj[j] - yi[j] - 0.5 * h * (fi[j] + fj[j])) / scales[j]
            idx += 1
        end
    end

    yT = node_slice(z, N)
    residual[idx] = (yT[1] - steady.k) / scales[1]
    idx += 1
    residual[idx] = (yT[2] - steady.c) / scales[2]
    idx += 1
    residual[idx] = (yT[3] - steady.q) / scales[3]
    idx += 1
    residual[idx] = (yT[5] - steady.Λ2) / scales[5]
end

function unpack_solution(z::AbstractVector, p::ModelParams, steady::SteadyStateResult, tgrid::AbstractVector, residual_norm::Real, success::Bool)
    N = length(tgrid)
    k = zeros(N)
    c = zeros(N)
    q = zeros(N)
    Λ1 = zeros(N)
    Λ2 = zeros(N)
    Λ3 = zeros(N)
    r_tilde = zeros(N)
    x = zeros(N)

    for i in 1:N
        yi = collect(node_slice(z, i))
        k[i], c[i], q[i], Λ1[i], Λ2[i], Λ3[i] = yi
        controls = foc_implied_controls(yi, p)
        if controls === nothing
            r_tilde[i] = NaN
            x[i] = NaN
        else
            r_tilde[i] = controls.r_tilde
            x[i] = controls.x
        end
    end

    return CollocationResult(success, collect(tgrid), k, c, q, Λ1, Λ2, Λ3, r_tilde, x, steady, Float64(residual_norm))
end

function solve_nonlinear_system(residual!, guess, p::ModelParams)
    try
        return nlsolve(residual!, guess; method = :trust_region, iterations = p.max_iter, xtol = 1e-10, ftol = 1e-10, show_trace = false)
    catch
        return nothing
    end
end

function solve_collocation_problem(p::ModelParams, steady::SteadyStateResult; N::Int = p.N, progress::Bool = true, initial_t = nothing, initial_z = nothing, use_mesh_continuation::Bool = true)
    stage_sizes = use_mesh_continuation ? unique(max.(11, [cld(N, 3), cld(2 * N, 3), N])) : [N]

    previous_t = initial_t
    previous_z = initial_z
    final_result = nothing

    for Ncur in stage_sizes
        tgrid = collocation_grid(p, Ncur)
        guess = previous_z === nothing ? collocation_guess(p, steady, tgrid) : interpolate_guess(previous_t, previous_z, tgrid)
        residual!(F, z) = collocation_residual!(F, z, p, steady, tgrid)
        progress && println("OptimalWealthTax collocation stage N=$(Ncur)")
        F = zeros(length(guess))
        nls = solve_nonlinear_system(residual!, guess, p)
        z = nls === nothing ? guess : nls.zero
        residual!(F, z)
        resnorm = maximum(abs.(F))
        success = isfinite(resnorm) && resnorm <= p.residual_tolerance
        previous_t = collect(tgrid)
        previous_z = copy(z)
        final_result = unpack_solution(previous_z, p, steady, previous_t, resnorm, success)

        if !success
            break
        end
    end

    return final_result, previous_t, previous_z
end

function continue_horizon(p::ModelParams, steady::SteadyStateResult; N::Int = p.N, progress::Bool = true, target_T::Real = p.T)
    base_T = ModelParams().T
    if target_T <= base_T + 1e-12
        return solve_collocation_problem(p, steady; N = N, progress = progress, use_mesh_continuation = true)
    end

    nstages = max(2, ceil(Int, (target_T - base_T) / base_T) + 1)
    T_stages = collect(range(base_T, Float64(target_T), length = nstages))
    previous_t = nothing
    previous_z = nothing
    final_result = nothing

    for T_stage in T_stages
        N_stage = max(11, round(Int, 1 + (N - 1) * T_stage / target_T))
        stage_params = with_horizon(p, T_stage, N_stage)
        stage_t = previous_t === nothing ? nothing : rescale_time_grid(previous_t, T_stage)
        progress && println("OptimalWealthTax horizon continuation T=$(round(T_stage; digits = 4)) N=$(N_stage)")
        final_result, previous_t, previous_z = solve_collocation_problem(stage_params, steady;
            N = N_stage,
            progress = progress,
            initial_t = stage_t,
            initial_z = previous_z,
            use_mesh_continuation = previous_z === nothing)
        if !final_result.success
            break
        end
    end

    return final_result, previous_t, previous_z
end

function solve_bvp_problem(p::ModelParams, steady::SteadyStateResult; N::Int = p.N, progress::Bool = true, initial_t, initial_z, terminal_mode::Symbol = :steady_state)
    tspan = (0.0, p.T)

    function f!(du, u, _, t)
        dy = dynamics(u, p)
        if all(isfinite, dy)
            du .= dy
        else
            du .= 0.0
        end
        return nothing
    end

    function bc!(res, u, _, t)
        ua = u[1]
        ub = u[end]
        res[1] = ua[1] - p.k0
        res[2] = ua[3] - p.q0
        if terminal_mode == :steady_state
            res[3] = ub[1] - steady.k
            res[4] = ub[2] - steady.c
            res[5] = ub[3] - steady.q
            res[6] = ub[5] - steady.Λ2
        elseif terminal_mode == :tvc
            controls = foc_implied_controls(ub, p)
            discount = exp(-p.ρ * p.T)
            res[3] = discount * ub[4] * ub[1]
            res[4] = discount * ub[5] * ub[2]
            res[5] = discount * ub[6] * ub[3]
            res[6] = controls === nothing ? 1e6 : controls.r_tilde - p.ρ
        else
            error("Unsupported terminal_mode=$(terminal_mode)")
        end
        return nothing
    end

    function guess_y(t)
        if initial_t === nothing || initial_z === nothing
            w = p.T <= 0 ? 0.0 : t / p.T
            return [
                (1.0 - w) * p.k0 + w * steady.k,
                steady.c,
                (1.0 - w) * p.q0 + w * steady.q,
                steady.Λ1,
                steady.Λ2,
                steady.Λ3,
            ]
        end
        return interpolate_state(initial_t, initial_z, t)
    end

    progress && println("OptimalWealthTax BVP refinement")
    prob = BVProblem(f!, bc!, guess_y, tspan)
    sol = solve(prob, MIRK6(); dt = p.T / max(N - 1, 1), abstol = 1e-8, reltol = 1e-8)

    tgrid = collocation_grid(p, N)
    z = zeros(6 * N)
    if SciMLBase.successful_retcode(sol.retcode)
        for (i, t) in enumerate(tgrid)
            yi = sol(t)
            offset = node_offset(i)
            z[offset + 1:offset + 6] .= yi
        end
    else
        z .= initial_z === nothing ? collocation_guess(p, steady, tgrid) : interpolate_guess(initial_t, initial_z, tgrid)
    end

    residual = zeros(length(z))
    collocation_residual!(residual, z, p, steady, tgrid)
    resnorm = maximum(abs.(residual))
    success = SciMLBase.successful_retcode(sol.retcode) && isfinite(resnorm) && resnorm <= p.residual_tolerance
    return unpack_solution(z, p, steady, tgrid, resnorm, success)
end

function continue_initial_conditions(p::ModelParams, steady::SteadyStateResult, previous_t, previous_z;
    N::Int = p.N,
    progress::Bool = true,
    base_step::Float64 = 0.025,
    min_step::Float64 = 1e-4,
    label::AbstractString,
    endpoint)
    current_alpha = 0.0
    final_result = unpack_solution(previous_z, p, steady, previous_t, 0.0, false)

    while current_alpha < 1.0 - 1e-12
        step = min(base_step, 1.0 - current_alpha)
        step_success = false

        while step >= min_step - 1e-12
            next_alpha = min(1.0, current_alpha + step)
            k0, q0 = endpoint(next_alpha)
            trial_params = with_initial_conditions(p, k0, q0)

            progress && println("OptimalWealthTax $(label) continuation α=$(round(next_alpha; digits = 4))")
            trial_result, trial_t, trial_z = solve_collocation_problem(trial_params, steady; N = N, progress = progress, initial_t = previous_t, initial_z = previous_z, use_mesh_continuation = false)

            if trial_result.success
                current_alpha = next_alpha
                previous_t = trial_t
                previous_z = trial_z
                final_result = trial_result
                step_success = true
                break
            end

            step *= 0.5
            if step >= min_step - 1e-12
                progress && println("  reducing continuation step to $(round(step; digits = 4))")
            end
        end

        if !step_success
            return current_alpha, final_result, previous_t, previous_z
        end
    end

    return current_alpha, final_result, previous_t, previous_z
end

function better_target_result(lhs::CollocationResult, rhs::CollocationResult)
    if lhs.success != rhs.success
        return lhs.success ? lhs : rhs
    end
    if isfinite(lhs.residual_norm) != isfinite(rhs.residual_norm)
        return isfinite(lhs.residual_norm) ? lhs : rhs
    end
    return lhs.residual_norm <= rhs.residual_norm ? lhs : rhs
end

function solve_collocation(p::ModelParams = ModelParams(); N::Int = p.N, progress::Bool = true, use_continuation::Bool = false, use_bvp_refinement::Bool = false, use_horizon_continuation::Bool = false)
    steady = find_steady_state(p)

    steady_params = with_initial_conditions(p, steady.k, steady.q)
    steady_result, previous_t, previous_z = solve_collocation_problem(steady_params, steady; N = N, progress = progress, use_mesh_continuation = true)
    if !steady_result.success
        return steady_result
    end

    target_gap = max(abs(p.k0 - steady.k) / max(abs(steady.k), 1.0), abs(p.q0 - steady.q) / max(abs(steady.q), 1.0))
    if target_gap <= 1e-12
        return steady_result
    end

    failed_target_attempts = CollocationResult[]

    direct_result, direct_t, direct_z = solve_collocation_problem(p, steady; N = N, progress = progress, use_mesh_continuation = true)
    if direct_result.success
        return direct_result
    end
    push!(failed_target_attempts, direct_result)

    if use_horizon_continuation
        horizon_result, horizon_t, horizon_z = continue_horizon(p, steady; N = N, progress = progress, target_T = p.T)
        if horizon_result.success
            return horizon_result
        end
        push!(failed_target_attempts, horizon_result)
        if better_target_result(horizon_result, direct_result) === horizon_result
            direct_t, direct_z = horizon_t, horizon_z
        end
    end

    if use_bvp_refinement
        push!(failed_target_attempts, solve_bvp_problem(p, steady; N = N, progress = progress, initial_t = direct_t, initial_z = direct_z))
    end

    if !use_continuation
        return reduce(better_target_result, failed_target_attempts)
    end

    diag_alpha, diag_result, previous_t, previous_z = continue_initial_conditions(p, steady, previous_t, previous_z;
        N = N,
        progress = progress,
        label = "diag",
        endpoint = α -> (steady.k + α * (p.k0 - steady.k), steady.q + α * (p.q0 - steady.q)))
    if diag_alpha >= 1.0 - 1e-12
        return diag_result
    end

    if use_bvp_refinement
        push!(failed_target_attempts, solve_bvp_problem(p, steady; N = N, progress = progress, initial_t = previous_t, initial_z = previous_z))
    end

    steady_result, previous_t, previous_z = solve_collocation_problem(steady_params, steady; N = N, progress = progress, use_mesh_continuation = true)
    if !steady_result.success
        return steady_result
    end

    q_alpha, q_result, previous_t, previous_z = continue_initial_conditions(p, steady, previous_t, previous_z;
        N = N,
        progress = progress,
        label = "q0",
        endpoint = α -> (steady.k, steady.q + α * (p.q0 - steady.q)))
    if q_alpha < 1.0 - 1e-12
        if use_bvp_refinement
            push!(failed_target_attempts, solve_bvp_problem(p, steady; N = N, progress = progress, initial_t = previous_t, initial_z = previous_z))
        end
        return reduce(better_target_result, failed_target_attempts)
    end

    k_alpha, k_result, previous_t, previous_z = continue_initial_conditions(p, steady, previous_t, previous_z;
        N = N,
        progress = progress,
        label = "k0",
        endpoint = α -> (steady.k + α * (p.k0 - steady.k), p.q0))
    if k_alpha < 1.0 - 1e-12
        if use_bvp_refinement
            push!(failed_target_attempts, solve_bvp_problem(p, steady; N = N, progress = progress, initial_t = previous_t, initial_z = previous_z))
        end
        return reduce(better_target_result, failed_target_attempts)
    end

    if k_result.success
        return k_result
    end

    if use_bvp_refinement
        push!(failed_target_attempts, solve_bvp_problem(p, steady; N = N, progress = progress, initial_t = previous_t, initial_z = previous_z))
    end
    return reduce(better_target_result, failed_target_attempts)
end