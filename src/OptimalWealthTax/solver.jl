"""
    CollocationResult

Struct storing the output of a collocation solve for the `OptimalWealthTax` model.

Fields:
- `success::Bool`: whether the residual tolerance was met.
- `t`, `k`, `c`, `q`, `Λ1`, `Λ2`, `Λ3`, `r_tilde`, `x`: vectors of length `N = length(t)`.
- `steady::SteadyStateResult`: scalar steady-state reference used by the solver.
- `residual_norm::Float64`: maximum absolute residual over the collocation system.
"""
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

"""
    print_progress_result(label, result)

Prints a compact progress summary for a collocation result.

Input arguments:
- `label::AbstractString`: stage or continuation label to print.
- `result::CollocationResult`: result to summarize.

Optional parameters:
- None.

Output:
- Returns `nothing`.
- Reads scalar diagnostics and the first/last entries of vector fields of length `N`.
"""
function print_progress_result(label::AbstractString, result::CollocationResult)
    println("OptimalWealthTax $(label): success=$(result.success) residual=$(result.residual_norm)")
    println("  init  k=$(result.k[1]) q=$(result.q[1]) Λ2=$(result.Λ2[1])")
    println("  final k=$(result.k[end]) c=$(result.c[end]) q=$(result.q[end]) Λ1=$(result.Λ1[end]) Λ2=$(result.Λ2[end]) Λ3=$(result.Λ3[end])")
    return nothing
end

canonical_terminal_mode(terminal_mode::Symbol) = terminal_mode == :tvc_lambda2c ? :tvc : terminal_mode
is_tvc_like_terminal_mode(terminal_mode::Symbol) = canonical_terminal_mode(terminal_mode) == :tvc

continuation_q_anchor(::ModelParams) = 0.0

function tvc_seed_terminal_mode(p::ModelParams)
    q_anchor = continuation_q_anchor(p)
    return abs(p.q0 - q_anchor) <= 1e-12 ? :state_steady_state_qtvc : :state_steady_state
end

function tvc_terminal_acceptance_tolerance(p::ModelParams)
    q_anchor = continuation_q_anchor(p)
    return abs(p.q0 - q_anchor) <= 1e-12 ? 6e-2 : 5e-2
end

function terminal_middle_metric(c, k, Λ2, p::ModelParams, terminal_mode::Symbol)
    terminal_mode = canonical_terminal_mode(terminal_mode)
    if terminal_mode == :tvc
        return max(c, p.min_positive)^(-p.β) * k
    else
        error("Unsupported TVC-like terminal_mode=$(terminal_mode)")
    end
end

"""
    transversality_metrics(result, p)

Computes discounted transversality diagnostics along a collocation path.

Input arguments:
- `result::CollocationResult`: solution path; all trajectory fields must have common length `N`.
- `p::ModelParams`: model parameters.

Optional parameters:
- None.

Output:
- Returns a named tuple with vector fields `k`, `c`, `q`, each of length `N`.
- The nested field `terminal` contains the final scalar values of those three diagnostics.
"""
function transversality_metrics(result::CollocationResult, p::ModelParams; terminal_mode::Symbol = :tvc)
    terminal_mode = canonical_terminal_mode(terminal_mode)
    discount = exp.(-p.ρ .* result.t)
    k_tvc = discount .* result.Λ1 .* result.k
    c_tvc = if terminal_mode == :tvc
        discount .* max.(result.c, p.min_positive) .^ (-p.β) .* result.k
    else
        error("Unsupported TVC-like terminal_mode=$(terminal_mode)")
    end
    q_tvc = discount .* result.Λ3 .* result.q
    terminal = (; k = k_tvc[end], c = c_tvc[end], q = q_tvc[end])
    return (; k = k_tvc, c = c_tvc, q = q_tvc, terminal)
end

"""
    control_reconstruction_diagnostics(y, p)

Computes local diagnostics for the reconstructed controls along a state-costate path.

Input arguments:
- `y::AbstractVector{<:Real}`: vector whose first five entries are `(k, c, q, Λ1, Λ2)`.
- `p::ModelParams`: model parameters.

Optional parameters:
- None.

Output:
- Returns a named tuple of scalar diagnostics describing the control reconstruction.
- The fields are `denom`, `r_unconstrained`, `x_unclamped`, `r_bound_active`, `x_floor_active`, and `denom_nonpositive`.
"""
function control_reconstruction_diagnostics(y::AbstractVector{<:Real}, p::ModelParams)
    base = control_reconstruction_terms(y, p)
    if base === nothing
        return (
            denom = NaN,
            r_unconstrained = NaN,
            x_unclamped = NaN,
            r_bound_active = false,
            x_floor_active = false,
            denom_nonpositive = false,
        )
    end

    r_unconstrained = base.r_unconstrained
    r_bound_active = isfinite(r_unconstrained) ? r_unconstrained <= 0.0 : r_unconstrained == -Inf
    x_unclamped = base.resource_term - base.sum_eff * max(0.0, r_unconstrained)
    x_floor_active = isfinite(x_unclamped) && x_unclamped <= p.min_positive
    denom_nonpositive = base.denom <= p.min_positive

    return (; denom = base.denom, r_unconstrained, x_unclamped, r_bound_active, x_floor_active, denom_nonpositive)
end

function infer_active_bound_masks(z::AbstractVector, p::ModelParams, tgrid::AbstractVector)
    N = length(tgrid)
    node_active = falses(N)
    midpoint_active = falses(max(N - 1, 0))
    activation_tolerance = max(100.0 * p.min_positive, 1e-8)

    for i in 1:N
        diagnostics = control_reconstruction_diagnostics(node_slice(z, i), p)
        node_active[i] = diagnostics.r_bound_active || (isfinite(diagnostics.r_unconstrained) && diagnostics.r_unconstrained <= activation_tolerance)
    end

    for i in 1:N-1
        yi = node_slice(z, i)
        yj = node_slice(z, i + 1)
        ymid = Vector{Float64}(undef, 6)
        for j in 1:6
            ymid[j] = 0.5 * (yi[j] + yj[j])
        end
        diagnostics = control_reconstruction_diagnostics(ymid, p)
        midpoint_active[i] = diagnostics.r_bound_active || (isfinite(diagnostics.r_unconstrained) && diagnostics.r_unconstrained <= activation_tolerance)
    end

    return (; node_active, midpoint_active)
end

function active_bound_masks_equal(lhs, rhs)
    return all(lhs.node_active .== rhs.node_active) && all(lhs.midpoint_active .== rhs.midpoint_active)
end

function should_try_active_set_stage(result::CollocationResult, p::ModelParams)
    if !isfinite(result.residual_norm) || isempty(result.t)
        return false
    end

    metrics = boundary_activity_metrics(result, p)
    return metrics.r_bound_fraction >= 0.2 ||
        metrics.denom_nonpositive_fraction >= 0.2 ||
        metrics.r_bound_terminal ||
        metrics.denom_nonpositive_terminal
end

"""
    boundary_activity_series(result, p)

Computes node-by-node diagnostics for control bounds and numerical clamps along a solution path.

Input arguments:
- `result::CollocationResult`: solution path with common trajectory length `N`.
- `p::ModelParams`: model parameters.

Optional parameters:
- None.

Output:
- Returns a named tuple of vectors, each of length `N`.
- The fields are `denom`, `r_unconstrained`, `x_unclamped`, `r_bound_active`, `x_floor_active`, and `denom_nonpositive`.
"""
function boundary_activity_series(result::CollocationResult, p::ModelParams)
    N = length(result.t)
    denom = fill(NaN, N)
    r_unconstrained = fill(NaN, N)
    x_unclamped = fill(NaN, N)
    r_bound_active = falses(N)
    x_floor_active = falses(N)
    denom_nonpositive = falses(N)

    for i in eachindex(result.t)
        diagnostics = control_reconstruction_diagnostics([
            result.k[i],
            result.c[i],
            result.q[i],
            result.Λ1[i],
            result.Λ2[i],
        ], p)
        denom[i] = diagnostics.denom
        r_unconstrained[i] = diagnostics.r_unconstrained
        x_unclamped[i] = diagnostics.x_unclamped
        r_bound_active[i] = diagnostics.r_bound_active
        x_floor_active[i] = diagnostics.x_floor_active
        denom_nonpositive[i] = diagnostics.denom_nonpositive
    end

    return (; denom, r_unconstrained, x_unclamped, r_bound_active, x_floor_active, denom_nonpositive)
end

function finite_minimum(values)
    best = Inf
    found = false
    for value in values
        if isfinite(value)
            best = min(best, value)
            found = true
        end
    end
    return found ? best : NaN
end

"""
    boundary_activity_metrics(result, p)

Aggregates diagnostics for boundary arcs and clamps along a solution path.

Input arguments:
- `result::CollocationResult`: solution path with common trajectory length `N`.
- `p::ModelParams`: model parameters.

Optional parameters:
- None.

Output:
- Returns a named tuple of scalar summary diagnostics.
- The fractions are measured over the `N` path nodes.
"""
function boundary_activity_metrics(result::CollocationResult, p::ModelParams)
    series = boundary_activity_series(result, p)
    N = max(length(result.t), 1)
    q_floor_threshold = max(100.0 * p.min_positive, 1e-8)

    return (
        r_bound_fraction = count(series.r_bound_active) / N,
        x_floor_fraction = count(series.x_floor_active) / N,
        denom_nonpositive_fraction = count(series.denom_nonpositive) / N,
        q_near_zero_fraction = count(q -> isfinite(q) && q <= q_floor_threshold, result.q) / N,
        r_bound_terminal = series.r_bound_active[end],
        x_floor_terminal = series.x_floor_active[end],
        denom_nonpositive_terminal = series.denom_nonpositive[end],
        min_denom = finite_minimum(series.denom),
        min_x_unclamped = finite_minimum(series.x_unclamped),
        min_q = finite_minimum(result.q),
    )
end

"""
    pack_solution(result)

Packs a structured collocation result into the flat vector layout used by the nonlinear solver.

Input arguments:
- `result::CollocationResult`: structured solution with trajectory length `N`.

Optional parameters:
- None.

Output:
- Returns a `Vector{Float64}` of length `6N`.
- The storage order at each node is `(k, c, q, Λ1, Λ2, Λ3)`.
"""
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

"""
    with_initial_conditions(p, k0, q0; Λ20=p.Λ20)

Builds a copy of `ModelParams` with updated initial conditions.

Input arguments:
- `p::ModelParams`: baseline parameter set.
- `k0::Real`: initial capital, scalar.
- `q0::Real`: initial auxiliary state, scalar.

Optional parameters:
- `Λ20::Real = p.Λ20`: initial value for `Λ2`.

Output:
- Returns a new `ModelParams` object.
- All stored quantities are scalars.
"""
function with_initial_conditions(p::ModelParams, k0::Real, q0::Real; Λ20::Real = p.Λ20)
    return ModelParams(A = p.A, θ = p.θ, η = p.η, β = p.β, ρ = p.ρ, δ = p.δ, γ = p.γ,
        n = p.n, l = p.l, k0 = Float64(k0), q0 = Float64(q0), Λ20 = Float64(Λ20), T = p.T, N = p.N,
    max_iter = p.max_iter, residual_tolerance = p.residual_tolerance, mesh_power = p.mesh_power,
    min_positive = p.min_positive)
end

"""
    with_horizon(p, T, N)

Builds a copy of `ModelParams` with a different horizon and mesh size.

Input arguments:
- `p::ModelParams`: baseline parameter set.
- `T::Real`: scalar time horizon.
- `N::Integer`: number of collocation nodes.

Optional parameters:
- None.

Output:
- Returns a new `ModelParams` object.
- All stored quantities are scalars.
"""
function with_horizon(p::ModelParams, T::Real, N::Integer)
    return ModelParams(A = p.A, θ = p.θ, η = p.η, β = p.β, ρ = p.ρ, δ = p.δ, γ = p.γ,
        n = p.n, l = p.l, k0 = p.k0, q0 = p.q0, Λ20 = p.Λ20, T = Float64(T), N = Int(N),
        max_iter = p.max_iter, residual_tolerance = p.residual_tolerance, mesh_power = p.mesh_power,
        min_positive = p.min_positive)
end

"""
    node_offset(i)

Returns the starting offset of node `i` inside the flat collocation vector.

Input arguments:
- `i::Int`: one-based node index.

Optional parameters:
- None.

Output:
- Returns an `Int` scalar.
- The output is the zero-based offset used before adding component positions `1:6`.
"""
node_offset(i::Int) = 6 * (i - 1)

"""
    node_slice(z, i)

Returns a view of the six variables stored at collocation node `i`.

Input arguments:
- `z::AbstractVector`: flat collocation vector of length `6N`.
- `i::Int`: one-based node index.

Optional parameters:
- None.

Output:
- Returns a vector view of length 6.
- The order is `(k, c, q, Λ1, Λ2, Λ3)`.
"""
function node_slice(z::AbstractVector, i::Int)
    offset = node_offset(i)
    return @view z[offset + 1:offset + 6]
end

"""
    collocation_guess_values(k, q, Λ2, steady, p)

Builds a local initial guess for the variables not fixed directly along a collocation seed path.

Input arguments:
- `k::Real`, `q::Real`, `Λ2::Real`: scalar values at a single node.
- `steady::SteadyStateResult`: steady-state reference.
- `p::ModelParams`: model parameters.

Optional parameters:
- None.

Output:
- Returns three scalars `(c_guess, Λ1_guess, Λ3_guess)`.
- Each returned quantity has size `1 x 1`.
"""
function collocation_guess_values(k::Real, q::Real, Λ2::Real, steady::SteadyStateResult, p::ModelParams)
    k_guess = max(Float64(k), p.min_positive)
    q_guess = Float64(q)
    Λ2_guess = Float64(Λ2)
    c_guess = steady.c
    sum_guess = max(k_guess + q_guess, p.min_positive)
    terms = production_terms(k_guess, p)

    # Keep the initial guess on a feasible branch by targeting the steady-state return.
    x_guess = terms.F - p.δ * k_guess - terms.Fn + q_guess * (terms.Fk - p.δ) - steady.r_tilde * sum_guess
    x_guess = (!isfinite(x_guess) || x_guess <= p.min_positive) ? steady.x : x_guess

    denom_target = p.γ * sum_guess / x_guess
    Λ1_guess = (denom_target - Λ2_guess * c_guess / p.β) / sum_guess
    Λ1_guess = isfinite(Λ1_guess) ? Λ1_guess : steady.Λ1

    Λ3_guess = Λ1_guess - p.γ / x_guess
    Λ3_guess = isfinite(Λ3_guess) ? Λ3_guess : steady.Λ3

    return c_guess, Λ1_guess, Λ3_guess
end

"""
    collocation_guess(p, steady, tgrid)

Constructs the flat initial guess used by the collocation solver on a given time grid.

Input arguments:
- `p::ModelParams`: model parameters.
- `steady::SteadyStateResult`: steady-state anchor.
- `tgrid::AbstractVector`: time grid of length `N`.

Optional parameters:
- None.

Output:
- Returns a `Vector{Float64}` of length `6N`.
- The vector stores `(k, c, q, Λ1, Λ2, Λ3)` at every node.
"""
function collocation_guess(p::ModelParams, steady::SteadyStateResult, tgrid::AbstractVector)
    N = length(tgrid)
    guess = zeros(6 * N)
    for (i, t) in enumerate(tgrid)
        w = tgrid[end] <= 0 ? 0.0 : t / tgrid[end]
        offset = node_offset(i)
        k_guess = (1.0 - w) * p.k0 + w * steady.k
        q_guess = (1.0 - w) * p.q0 + w * steady.q
        Λ2_guess = (1.0 - w) * p.Λ20 + w * steady.Λ2
        c_guess, Λ1_guess, Λ3_guess = collocation_guess_values(k_guess, q_guess, Λ2_guess, steady, p)

        guess[offset + 1] = k_guess
        guess[offset + 2] = c_guess
        guess[offset + 3] = q_guess
        guess[offset + 4] = Λ1_guess
        guess[offset + 5] = Λ2_guess
        guess[offset + 6] = Λ3_guess
    end
    return guess
end

"""
    interpolate_guess(old_t, old_z, new_t)

Interpolates a flat collocation vector from one time grid to another component by component.

Input arguments:
- `old_t::AbstractVector`: original time grid of length `N_old`.
- `old_z::AbstractVector`: flat solution vector of length `6N_old`.
- `new_t::AbstractVector`: target time grid of length `N_new`.

Optional parameters:
- None.

Output:
- Returns a `Vector{Float64}` of length `6N_new`.
- The component ordering is preserved node by node.
"""
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

"""
    interpolate_state(old_t, old_z, t)

Interpolates the collocation state-costate vector at a single time point.

Input arguments:
- `old_t::AbstractVector`: original time grid of length `N`.
- `old_z::AbstractVector`: flat solution vector of length `6N`.
- `t::Real`: target time.

Optional parameters:
- None.

Output:
- Returns a `Vector{Float64}` of length 6.
- The returned order is `(k, c, q, Λ1, Λ2, Λ3)`.
"""
function interpolate_state(old_t::AbstractVector, old_z::AbstractVector, t::Real)
    tmp = interpolate_guess(old_t, old_z, [Float64(t)])
    return collect(node_slice(tmp, 1))
end

"""
    rescale_time_grid(old_t, new_T)

Rescales an existing time grid so that its final node becomes `new_T`.

Input arguments:
- `old_t::AbstractVector`: original time grid of length `N`.
- `new_T::Real`: new terminal time.

Optional parameters:
- None.

Output:
- Returns a `Vector{Float64}` of length `N`.
- If the original terminal time is nonpositive, it returns a zero vector of length `N`.
"""
function rescale_time_grid(old_t::AbstractVector, new_T::Real)
    old_T = old_t[end]
    if old_T <= 0
        return zeros(length(old_t))
    end
    return collect(Float64(new_T) .* (old_t ./ old_T))
end

"""
    collocation_grid(p, N)

Builds the front-loaded time grid used by collocation.

Input arguments:
- `p::ModelParams`: model parameters.
- `N::Int`: number of grid nodes.

Optional parameters:
- None.

Output:
- Returns a `Vector{Float64}` of length `N`.
- The first entry is `0.0` and the last entry is `p.T`.
"""
function collocation_grid(p::ModelParams, N::Int)
    ξ = range(0.0, 1.0, length = N)
    return collect(p.T .* (ξ .^ p.mesh_power))
end

"""
    terminal_residual_values(yT, p, steady, scales, terminal_time, terminal_mode)

Evaluates the terminal residual block associated with a chosen closure rule.

Input arguments:
- `yT::AbstractVector`: terminal node values; expected length is 6.
- `p::ModelParams`: model parameters.
- `steady::SteadyStateResult`: steady-state reference.
- `scales`: tuple of six scalar normalization factors.
- `terminal_time::Real`: terminal time, scalar.
- `terminal_mode::Symbol`: one of `:steady_state`, `:costate_steady_state`, `:state_steady_state`, or `:tvc`.

Optional parameters:
- None.

Output:
- Returns a `Vector{Float64}` of length 3.
- The meaning of the three entries depends on `terminal_mode`.
"""
function terminal_residual_values(yT::AbstractVector, p::ModelParams, steady::SteadyStateResult, scales, terminal_time::Real, terminal_mode::Symbol)
    terminal_mode = canonical_terminal_mode(terminal_mode)
    if terminal_mode == :steady_state || terminal_mode == :costate_steady_state
        return [
            (yT[2] - steady.c) / scales[2],
            (yT[4] - steady.Λ1) / scales[4],
            (yT[6] - steady.Λ3) / scales[6],
        ]
    elseif terminal_mode == :state_steady_state
        return [
            (yT[1] - steady.k) / scales[1],
            (yT[3] - steady.q) / scales[3],
            (yT[2] - steady.c) / scales[2],
        ]
    elseif terminal_mode == :state_stationary
        controls = foc_implied_controls(yT, p)
        if controls === nothing
            return fill(1e6, 3)
        end
        return [
            (yT[1] - steady.k) / scales[1],
            (controls.r_tilde - p.ρ) / max(abs(p.ρ), 1.0),
            (yT[3] - steady.q) / scales[3],
        ]
    elseif terminal_mode == :state_steady_state_qtvc
        discount = exp(-p.ρ * terminal_time)
        return [
            (yT[1] - steady.k) / scales[1],
            discount * yT[6] * yT[3],
            (yT[2] - steady.c) / scales[2],
        ]
    elseif is_tvc_like_terminal_mode(terminal_mode)
        discount = exp(-p.ρ * terminal_time)
        return [
            discount * yT[4] * yT[1],
            discount * terminal_middle_metric(yT[2], yT[1], yT[5], p, terminal_mode),
            discount * yT[6] * yT[3],
        ]
    else
        error("Unsupported terminal_mode=$(terminal_mode)")
    end
end

"""
    terminal_residuals!(residual, idx, yT, p, steady, scales, terminal_time, terminal_mode; terminal_alpha=1.0)

Writes the terminal residual block in place into a larger residual vector.

Input arguments:
- `residual`: residual vector being assembled.
- `idx::Int`: starting index of the terminal block.
- `yT::AbstractVector`: terminal node values of length 6.
- `p::ModelParams`: model parameters.
- `steady::SteadyStateResult`: steady-state reference.
- `scales`: tuple of six scalar normalization factors.
- `terminal_time::Real`: terminal time.
- `terminal_mode::Symbol`: terminal closure mode.

Optional parameters:
- `terminal_alpha::Float64=1.0`: homotopy weight between state-steady-state closure and TVC closure when `terminal_mode == :tvc`.

Output:
- Returns `nothing`.
- Overwrites three consecutive entries of `residual` starting at `idx`.
"""
function terminal_residuals!(residual, idx::Int, yT::AbstractVector, p::ModelParams, steady::SteadyStateResult, scales, terminal_time::Real, terminal_mode::Symbol; terminal_alpha::Float64 = 1.0)
    values = if terminal_alpha < 1.0 - 1e-12 && is_tvc_like_terminal_mode(terminal_mode)
        (1.0 - terminal_alpha) .* terminal_residual_values(yT, p, steady, scales, terminal_time, tvc_seed_terminal_mode(p)) .+
        terminal_alpha .* terminal_residual_values(yT, p, steady, scales, terminal_time, terminal_mode)
    elseif terminal_alpha < 1.0 - 1e-12 && terminal_mode == :state_steady_state
        (1.0 - terminal_alpha) .* terminal_residual_values(yT, p, steady, scales, terminal_time, :costate_steady_state) .+
        terminal_alpha .* terminal_residual_values(yT, p, steady, scales, terminal_time, :state_steady_state)
    else
        terminal_residual_values(yT, p, steady, scales, terminal_time, terminal_mode)
    end
    residual[idx:idx + 2] .= values
    return nothing
end

"""
    collocation_residual!(residual, z, p, steady, tgrid; terminal_mode=:steady_state, terminal_alpha=1.0)

Assembles the full collocation residual vector in place.

Input arguments:
- `residual`: preallocated residual vector of length `6N`.
- `z`: flat unknown vector of length `6N`.
- `p::ModelParams`: model parameters.
- `steady::SteadyStateResult`: steady-state reference.
- `tgrid::AbstractVector`: time grid of length `N`.

Optional parameters:
- `terminal_mode::Symbol = :steady_state`: terminal closure rule.
- `terminal_alpha::Float64 = 1.0`: homotopy weight used by TVC continuation.

Output:
- Returns `nothing`.
- Fills `residual` with three initial-condition equations, `6(N-1)` defect equations, and three terminal equations.
"""
function collocation_residual!(residual, z, p::ModelParams, steady::SteadyStateResult, tgrid::AbstractVector;
    terminal_mode::Symbol = :steady_state,
    terminal_alpha::Float64 = 1.0,
    active_bound_nodes = nothing,
    active_bound_midpoints = nothing)
    N = length(tgrid)
    fill!(residual, 0.0)
    idx = 1
    initial_scales = (
        max(abs(p.k0), 1.0),
        max(abs(p.q0), 1.0),
        max(abs(p.Λ20), 1.0),
    )
    scales = (
        max(abs(p.k0), abs(steady.k), 1.0),
        max(abs(steady.c), 1.0),
        max(abs(p.q0), abs(steady.q), 1.0),
        max(abs(steady.Λ1), 1.0),
        max(abs(p.Λ20), abs(steady.Λ2), 1.0),
        max(abs(steady.Λ3), 1.0),
    )

    y0 = node_slice(z, 1)
    residual[idx] = (y0[1] - p.k0) / initial_scales[1]
    idx += 1
    residual[idx] = (y0[3] - p.q0) / initial_scales[2]
    idx += 1
    residual[idx] = (y0[5] - p.Λ20) / initial_scales[3]
    idx += 1
    for i in 1:N-1
        yi = collect(node_slice(z, i))
        yj = collect(node_slice(z, i + 1))
        fi = dynamics(yi, p; active_bound = active_bound_nodes === nothing ? nothing : active_bound_nodes[i])
        fj = dynamics(yj, p; active_bound = active_bound_nodes === nothing ? nothing : active_bound_nodes[i + 1])
        h = tgrid[i + 1] - tgrid[i]
        if !(all(isfinite, fi) && all(isfinite, fj))
            residual[idx:idx + 5] .= 1e6
            idx += 6
            continue
        end
        ymid = similar(yi)
        for j in 1:6
            ymid[j] = 0.5 * (yi[j] + yj[j]) - 0.125 * h * (fj[j] - fi[j])
        end
        fmid = dynamics(ymid, p; active_bound = active_bound_midpoints === nothing ? nothing : active_bound_midpoints[i])
        if !all(isfinite, fmid)
            residual[idx:idx + 5] .= 1e6
            idx += 6
            continue
        end
        for j in 1:6
            residual[idx] = (yj[j] - yi[j] - (h / 6.0) * (fi[j] + 4.0 * fmid[j] + fj[j])) / scales[j]
            idx += 1
        end
    end

    yT = node_slice(z, N)
    terminal_residuals!(residual, idx, yT, p, steady, scales, tgrid[end], terminal_mode; terminal_alpha = terminal_alpha)
end

function state_path_is_admissible(y::AbstractVector, p::ModelParams)
    return all(isfinite, y) && y[1] > p.min_positive && y[2] > p.min_positive && y[3] >= 0.0
end

function solution_path_is_admissible(z::AbstractVector, p::ModelParams)
    for node in 1:div(length(z), 6)
        if !state_path_is_admissible(node_slice(z, node), p)
            return false
        end
    end
    return true
end

"""
    unpack_solution(z, p, steady, tgrid, residual_norm, success)

Converts a flat collocation vector into a structured `CollocationResult`.

Input arguments:
- `z::AbstractVector`: flat vector of length `6N`.
- `p::ModelParams`: model parameters.
- `steady::SteadyStateResult`: steady-state reference.
- `tgrid::AbstractVector`: time grid of length `N`.
- `residual_norm::Real`: scalar residual summary.
- `success::Bool`: success flag.

Optional parameters:
- None.

Output:
- Returns a `CollocationResult`.
- All trajectory fields in the result have length `N`.
"""
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

    path_is_admissible = solution_path_is_admissible(z, p)
    effective_success = success && path_is_admissible
    effective_residual_norm = path_is_admissible ? Float64(residual_norm) : Inf
    return CollocationResult(effective_success, collect(tgrid), k, c, q, Λ1, Λ2, Λ3, r_tilde, x, steady, effective_residual_norm)
end

"""
    evaluate_candidate(z, p, steady, tgrid; success=false, terminal_mode=:steady_state, terminal_alpha=1.0)

Evaluates a flat candidate path by recomputing its residual norm and unpacking it as a structured result.

Input arguments:
- `z::AbstractVector`: flat vector of length `6N`.
- `p::ModelParams`: model parameters.
- `steady::SteadyStateResult`: steady-state reference.
- `tgrid::AbstractVector`: time grid of length `N`.

Optional parameters:
- `success::Bool=false`: requested success flag before residual verification.
- `terminal_mode::Symbol=:steady_state`: terminal closure rule.
- `terminal_alpha::Float64=1.0`: homotopy weight for TVC continuation.

Output:
- Returns a `CollocationResult` with trajectory length `N`.
- The final `success` flag is true only if the supplied flag is true and the residual norm is below tolerance.
"""
function evaluate_candidate(z::AbstractVector, p::ModelParams, steady::SteadyStateResult, tgrid::AbstractVector; success::Bool = false, terminal_mode::Symbol = :steady_state, terminal_alpha::Float64 = 1.0)
    residual = zeros(length(z))
    collocation_residual!(residual, z, p, steady, tgrid; terminal_mode = terminal_mode, terminal_alpha = terminal_alpha)
    resnorm = maximum(abs.(residual))
    actual_success = success && isfinite(resnorm) && resnorm <= p.residual_tolerance
    return unpack_solution(z, p, steady, tgrid, resnorm, actual_success)
end

function solve_active_set_stage(p::ModelParams, steady::SteadyStateResult, tgrid::AbstractVector, guess::AbstractVector;
    progress::Bool = true,
    terminal_mode::Symbol = :steady_state,
    terminal_alpha::Float64 = 1.0,
    max_active_set_iterations::Int = 4)
    current_z = copy(guess)
    best_result = evaluate_candidate(current_z, p, steady, tgrid;
        success = false,
        terminal_mode = terminal_mode,
        terminal_alpha = terminal_alpha)
    best_z = copy(current_z)
    masks = infer_active_bound_masks(current_z, p, tgrid)

    for iter in 1:max_active_set_iterations
        residual!(F, z) = collocation_residual!(F, z, p, steady, tgrid;
            terminal_mode = terminal_mode,
            terminal_alpha = terminal_alpha,
            active_bound_nodes = masks.node_active,
            active_bound_midpoints = masks.midpoint_active)
        F = zeros(length(current_z))
        residual!(F, current_z)
        progress && println("  active-set iter=$(iter) masked_residual=$(maximum(abs.(F))) active_nodes=$(count(masks.node_active)) active_midpoints=$(count(masks.midpoint_active))")

        nls = solve_nonlinear_system(residual!, current_z, p; show_trace = progress)
        trial_z = nls === nothing ? current_z : nls.zero
        candidate = evaluate_candidate(trial_z, p, steady, tgrid;
            success = false,
            terminal_mode = terminal_mode,
            terminal_alpha = terminal_alpha)
        progress && println("  active-set iter=$(iter) true_residual=$(candidate.residual_norm)")

        if better_target_result(candidate, best_result) === candidate
            best_result = candidate
            best_z = copy(trial_z)
        end

        new_masks = infer_active_bound_masks(trial_z, p, tgrid)
        current_z = copy(trial_z)
        if active_bound_masks_equal(masks, new_masks)
            break
        end
        masks = new_masks
    end

    return best_result, best_z
end

"""
    solve_nonlinear_system(residual!, guess, p; show_trace=false)

Runs an Ipopt feasibility solve on the collocation residual system with the solver settings stored in `p`.

Input arguments:
- `residual!`: in-place residual callback.
- `guess`: initial guess vector, typically length `6N`.
- `p::ModelParams`: model parameters providing iteration limits.

Optional parameters:
- `show_trace::Bool = false`: whether to print Ipopt iteration progress.

Output:
- Returns a named tuple containing the optimizer status and solution vector.
- Returns `nothing` if the nonlinear solve throws an exception.
"""
function collocation_objective(residual!, z::AbstractVector)
    residual = similar(z)
    residual!(residual, z)
    total = zero(eltype(z))
    @inbounds for value in residual
        total += value * value
    end
    return total
end

function solve_nonlinear_system(residual!, guess, p::ModelParams; show_trace::Bool = false)
    try
        n = length(guess)
        m = n
        x_L = fill(-Inf, n)
        x_U = fill(Inf, n)
        for node in 1:div(n, 6)
            x_L[node_offset(node) + 1] = p.min_positive
            x_L[node_offset(node) + 2] = p.min_positive
            x_L[node_offset(node) + 3] = 0.0
        end

        residual_map = let residual! = residual!
            function (x)
                values = similar(x)
                residual!(values, x)
                return values
            end
        end

        eval_f(x) = 0.5e-8 * sum(abs2, x .- guess)
        function eval_g(x, g)
            residual!(g, x)
            return nothing
        end
        function eval_grad_f(x, grad_f)
            @inbounds for i in eachindex(grad_f)
                grad_f[i] = 1e-8 * (x[i] - guess[i])
            end
            return nothing
        end
        function eval_jac_g(x, rows, cols, values)
            if values === nothing
                idx = 1
                for col in 1:n
                    for row in 1:m
                        rows[idx] = row
                        cols[idx] = col
                        idx += 1
                    end
                end
            else
                values .= vec(ForwardDiff.jacobian(residual_map, x))
            end
            return nothing
        end

        problem = Ipopt.CreateIpoptProblem(
            n,
            x_L,
            x_U,
            m,
            zeros(m),
            zeros(m),
            m * n,
            0,
            eval_f,
            eval_g,
            eval_grad_f,
            eval_jac_g,
            nothing,
        )
        problem.x .= guess

        Ipopt.AddIpoptIntOption(problem, "max_iter", p.max_iter)
        Ipopt.AddIpoptNumOption(problem, "tol", 1e-10)
        Ipopt.AddIpoptNumOption(problem, "acceptable_tol", max(1e-8, p.residual_tolerance))
        Ipopt.AddIpoptIntOption(problem, "acceptable_iter", 5)
        Ipopt.AddIpoptStrOption(problem, "hessian_approximation", "limited-memory")
        Ipopt.AddIpoptNumOption(problem, "constr_viol_tol", p.residual_tolerance)
        Ipopt.AddIpoptIntOption(problem, "print_level", show_trace ? 5 : 0)
        if !show_trace
            Ipopt.AddIpoptStrOption(problem, "sb", "yes")
        end

        Ipopt.IpoptSolve(problem)
        return (; zero = copy(problem.x), status = problem.status, objective = problem.obj_val)
    catch err
        if show_trace
            println("OptimalWealthTax Ipopt solve setup failed: $(sprint(showerror, err))")
        end
        return nothing
    end
end

"""
    shooting_unknowns_from_result(result, p)

Builds the shooting unknown vector `(log(c0), Λ10, Λ30)` from a structured solution path.

Input arguments:
- `result::CollocationResult`: reference path.
- `p::ModelParams`: model parameters.

Optional parameters:
- None.

Output:
- Returns a `Vector{Float64}` of length 3.
"""
function shooting_unknowns_from_result(result::CollocationResult, p::ModelParams)
    return [
        log(max(result.c[1], p.min_positive)),
        result.Λ1[1],
        result.Λ3[1],
    ]
end

"""
    shooting_initial_state(v, p)

Reconstructs the initial state-costate vector from the shooting unknowns.

Input arguments:
- `v::AbstractVector`: vector of length 3 storing `(log(c0), Λ10, Λ30)`.
- `p::ModelParams`: model parameters.

Optional parameters:
- None.

Output:
- Returns a `Vector{Float64}` of length 6 ordered as `(k0, c0, q0, Λ10, Λ20, Λ30)`.
"""
function shooting_initial_state(v::AbstractVector, p::ModelParams)
    return [
        p.k0,
        exp(v[1]),
        p.q0,
        v[2],
        p.Λ20,
        v[3],
    ]
end

"""
    shooting_scales(p, steady)

Builds the component-wise normalization scales used by the terminal residual block.

Input arguments:
- `p::ModelParams`: model parameters.
- `steady::SteadyStateResult`: steady-state reference.

Optional parameters:
- None.

Output:
- Returns a tuple of six scalar normalization factors.
"""
function shooting_scales(p::ModelParams, steady::SteadyStateResult)
    return (
        max(abs(p.k0), abs(steady.k), 1.0),
        max(abs(steady.c), 1.0),
        max(abs(p.q0), abs(steady.q), 1.0),
        max(abs(steady.Λ1), 1.0),
        max(abs(p.Λ20), abs(steady.Λ2), 1.0),
        max(abs(steady.Λ3), 1.0),
    )
end

"""
    solve_shooting_ivp(p, initial_state; saveat=nothing)

Integrates the optimal-tax state-costate dynamics forward from a fully specified initial state.

Input arguments:
- `p::ModelParams`: model parameters.
- `initial_state::AbstractVector`: vector of length 6 ordered as `(k, c, q, Λ1, Λ2, Λ3)`.

Optional parameters:
- `saveat=nothing`: optional output grid.

Output:
- Returns the ODE solution object on success.
- Returns `nothing` if the IVP cannot be solved.
"""
function solve_shooting_ivp(p::ModelParams, initial_state::AbstractVector; saveat = nothing)
    function f!(du, u, _, t)
        dy = dynamics(u, p)
        du .= dy
        return nothing
    end

    function terminate_condition(u, t, integrator)
        return !state_path_is_admissible(u, p)
    end

    function terminate_affect!(integrator)
        terminate!(integrator)
        return nothing
    end

    problem = ODEProblem(f!, collect(initial_state), (0.0, p.T))
    terminate_callback = DiscreteCallback(terminate_condition, terminate_affect!)
    common_kwargs = (
        abstol = 1e-8,
        reltol = 1e-8,
        dt = min(1e-3, max(p.T / 2000, 1e-6)),
        dtmin = 1e-12,
        dtmax = max(1e-2, p.T / 200),
        maxiters = 20_000_000,
        callback = terminate_callback,
    )
    try
        if saveat === nothing
            return solve(problem, TRBDF2(); common_kwargs..., save_everystep = false)
        end
        return solve(problem, TRBDF2(); common_kwargs..., saveat = saveat)
    catch
        return nothing
    end
end

"""
    shooting_terminal_residuals(v, p, steady; terminal_mode=:tvc)

Evaluates the terminal residual block induced by a shooting guess.

Input arguments:
- `v::AbstractVector`: vector of length 3 storing `(log(c0), Λ10, Λ30)`.
- `p::ModelParams`: model parameters.
- `steady::SteadyStateResult`: steady-state reference.

Optional parameters:
- `terminal_mode::Symbol = :tvc`: terminal closure rule.

Output:
- Returns a tuple `(residuals, sol)` where `residuals` is a vector of length 3.
- If the IVP fails, the residual vector is filled with `1e6` and `sol` is `nothing`.
"""
function shooting_terminal_residuals(v::AbstractVector, p::ModelParams, steady::SteadyStateResult; terminal_mode::Symbol = :tvc)
    initial_state = shooting_initial_state(v, p)
    if !state_path_is_admissible(initial_state, p)
        return fill(1e6, 3), nothing
    end

    sol = solve_shooting_ivp(p, initial_state)
    if sol === nothing || sol.retcode != SciMLBase.ReturnCode.Success || isempty(sol.u)
        return fill(1e6, 3), sol
    end

    yT = sol.u[end]
    if !state_path_is_admissible(yT, p)
        return fill(1e6, 3), sol
    end

    residuals = terminal_residual_values(yT, p, steady, shooting_scales(p, steady), p.T, terminal_mode)
    if !all(isfinite, residuals)
        return fill(1e6, 3), sol
    end
    return residuals, sol
end

"""
    shooting_result_from_solution(sol, root, p, steady; output_N=max(p.N, 201), terminal_mode=:tvc)

Projects a shooting IVP solution to a common output grid and packages it as a `CollocationResult`.

Input arguments:
- `sol`: IVP solution object.
- `root`: terminal residual vector.
- `p::ModelParams`: model parameters.
- `steady::SteadyStateResult`: steady-state reference.

Optional parameters:
- `output_N::Int = max(p.N, 201)`: output grid size.
- `terminal_mode::Symbol = :tvc`: terminal closure rule used for success reporting.

Output:
- Returns a `CollocationResult` whose `residual_norm` is the maximum absolute terminal residual.
"""
function shooting_result_from_solution(sol, root::AbstractVector, p::ModelParams, steady::SteadyStateResult;
    output_N::Int = max(p.N, 201),
    terminal_mode::Symbol = :tvc)
    terminal_mode = canonical_terminal_mode(terminal_mode)
    tgrid = collect(range(0.0, p.T, length = output_N))
    z = zeros(6 * output_N)
    for (i, t) in enumerate(tgrid)
        yi = sol(t)
        z[node_offset(i) + 1:node_offset(i) + 6] .= yi
    end
    resnorm = maximum(abs.(root))
    success = isfinite(resnorm) && resnorm <= p.residual_tolerance && terminal_mode == :tvc
    return unpack_solution(z, p, steady, tgrid, resnorm, success)
end

"""
    shooting_seed_candidates(initial_guess, steady, p)

Builds a small set of shooting seeds around the current and steady-state anchors.

Input arguments:
- `initial_guess`: optional vector of length 3.
- `steady::SteadyStateResult`: steady-state reference.
- `p::ModelParams`: model parameters.

Optional parameters:
- None.

Output:
- Returns a vector of vectors, each of length 3.
"""
function shooting_seed_candidates(initial_guess, steady::SteadyStateResult, p::ModelParams)
    seeds = Vector{Vector{Float64}}()
    steady_guess = [log(max(steady.c, p.min_positive)), steady.Λ1, steady.Λ3]
    if initial_guess !== nothing
        current = collect(initial_guess)
        push!(seeds, current)
        c_level = exp(current[1])
        λ1_scale = max(abs(current[2]), 1.0)
        λ3_scale = max(abs(current[3]), 1.0)
        for c_mult in (0.95, 1.05)
            push!(seeds, [log(max(c_mult * c_level, p.min_positive)), current[2], current[3]])
        end
        push!(seeds, [current[1], current[2] + 0.1 * λ1_scale, current[3]])
        push!(seeds, [current[1], current[2] - 0.1 * λ1_scale, current[3]])
        push!(seeds, [current[1], current[2], current[3] + 0.1 * λ3_scale])
        push!(seeds, [current[1], current[2], current[3] - 0.1 * λ3_scale])
    end
    push!(seeds, steady_guess)

    λ1_scale = max(abs(steady.Λ1), 1.0)
    λ3_scale = max(abs(steady.Λ3), 1.0)
    for c_mult in (0.9, 1.1)
        push!(seeds, [log(max(c_mult * steady.c, p.min_positive)), steady.Λ1, steady.Λ3])
    end
    push!(seeds, [steady_guess[1], steady.Λ1 + 0.25 * λ1_scale, steady.Λ3])
    push!(seeds, [steady_guess[1], steady.Λ1 - 0.25 * λ1_scale, steady.Λ3])
    push!(seeds, [steady_guess[1], steady.Λ1, steady.Λ3 + 0.25 * λ3_scale])
    push!(seeds, [steady_guess[1], steady.Λ1, steady.Λ3 - 0.25 * λ3_scale])
    return seeds
end

"""
    solve_shooting_stage(p, steady; progress=true, initial_guess=nothing, terminal_mode=:tvc, output_N=max(p.N, 201))

Solves one forward-shooting stage by root-finding over `(log(c0), Λ10, Λ30)`.

Input arguments:
- `p::ModelParams`: model parameters.
- `steady::SteadyStateResult`: steady-state reference.

Optional parameters:
- `progress::Bool = true`: print stage progress.
- `initial_guess=nothing`: optional vector of length 3.
- `terminal_mode::Symbol = :tvc`: terminal closure rule.
- `output_N::Int = max(p.N, 201)`: output grid size for the returned path.

Output:
- Returns a tuple `(best_result, best_guess)`.
"""
function solve_shooting_stage(p::ModelParams, steady::SteadyStateResult;
    progress::Bool = true,
    initial_guess = nothing,
    terminal_mode::Symbol = :tvc,
    output_N::Int = max(p.N, 201))
    best_result = nothing
    best_guess = nothing
    best_residual = Inf

    for (seed_idx, seed) in enumerate(shooting_seed_candidates(initial_guess, steady, p))
        local_residual!(F, v) = begin
            residuals, _ = shooting_terminal_residuals(v, p, steady; terminal_mode = terminal_mode)
            F .= residuals
            return F
        end

        stage_solver = try
            nlsolve(local_residual!, seed;
                method = :trust_region,
                iterations = p.max_iter,
                xtol = 1e-10,
                ftol = 1e-10,
                show_trace = false)
        catch
            nothing
        end
        if stage_solver === nothing
            continue
        end

        root, _ = shooting_terminal_residuals(stage_solver.zero, p, steady; terminal_mode = terminal_mode)
        root_norm = maximum(abs.(root))
        if progress
            println("OptimalWealthTax shooting seed #$(seed_idx): residual=$(root_norm) c0=$(exp(stage_solver.zero[1])) Λ10=$(stage_solver.zero[2]) Λ30=$(stage_solver.zero[3])")
        end
        if !(isfinite(root_norm) && root_norm < best_residual)
            continue
        end

        _, final_sol = shooting_terminal_residuals(stage_solver.zero, p, steady; terminal_mode = terminal_mode)
        if final_sol === nothing || final_sol.retcode != SciMLBase.ReturnCode.Success || isempty(final_sol.u)
            continue
        end

        best_residual = root_norm
        best_guess = collect(stage_solver.zero)
        best_result = shooting_result_from_solution(final_sol, root, p, steady;
            output_N = output_N,
            terminal_mode = terminal_mode)
    end

    if best_result === nothing
        fallback_guess = initial_guess === nothing ? [log(max(steady.c, p.min_positive)), steady.Λ1, steady.Λ3] : collect(initial_guess)
        fallback_root, fallback_sol = shooting_terminal_residuals(fallback_guess, p, steady; terminal_mode = terminal_mode)
        if fallback_sol !== nothing && fallback_sol.retcode == SciMLBase.ReturnCode.Success && !isempty(fallback_sol.u)
            return shooting_result_from_solution(fallback_sol, fallback_root, p, steady;
                output_N = output_N,
                terminal_mode = terminal_mode), fallback_guess
        end
        tgrid = collect(range(0.0, p.T, length = output_N))
        z = repeat(shooting_initial_state(fallback_guess, p), output_N)
        return unpack_solution(z, p, steady, tgrid, 1e6, false), fallback_guess
    end

    return best_result, best_guess
end

"""
    default_shooting_horizon_stages(target_T)

Builds a conservative list of horizon-continuation stages for the shooting solver.

Input arguments:
- `target_T::Real`: target horizon.

Optional parameters:
- None.

Output:
- Returns a sorted `Vector{Float64}`.
"""
function default_shooting_horizon_stages(target_T::Real)
    stages = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0,
        12.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0,
        120.0, 160.0, Float64(target_T),
    ]
    filtered = sort(unique(t for t in stages if t > 0 && t <= target_T + 1e-12))
    return isempty(filtered) ? [Float64(target_T)] : filtered
end

"""
    solve_shooting(p=ModelParams(); progress=true, terminal_mode=:tvc, use_horizon_continuation=true, output_N=max(p.N, 201), initial_guess=nothing)

High-level forward-shooting solver for the `OptimalWealthTax` model.

Input arguments:
- `p::ModelParams = ModelParams()`: model parameters.

Optional parameters:
- `progress::Bool = true`: print progress messages.
- `terminal_mode::Symbol = :tvc`: terminal closure rule.
- `use_horizon_continuation::Bool = true`: solve intermediate horizons before the target horizon.
- `output_N::Int = max(p.N, 201)`: output grid size for the returned path.
- `initial_guess=nothing`: optional vector of length 3 storing `(log(c0), Λ10, Λ30)`.

Output:
- Returns a `CollocationResult` built from the best shooting path found.
"""
function solve_shooting(p::ModelParams = ModelParams();
    progress::Bool = true,
    terminal_mode::Symbol = :tvc,
    use_horizon_continuation::Bool = true,
    output_N::Int = max(p.N, 201),
    initial_guess = nothing)
    steady = find_steady_state(p)
    T_stages = use_horizon_continuation ? default_shooting_horizon_stages(p.T) : [p.T]

    progress && println("OptimalWealthTax shooting solve start: terminal_mode=$(terminal_mode) T=$(p.T) k0=$(p.k0) q0=$(p.q0) Λ20=$(p.Λ20)")
    current_guess = initial_guess
    target_result = nothing
    last_stage_result = nothing
    for stage_T in T_stages
        stage_params = with_horizon(p, stage_T, p.N)
        progress && println("OptimalWealthTax shooting stage T=$(stage_T)")
        stage_result, current_guess = solve_shooting_stage(stage_params, steady;
            progress = progress,
            initial_guess = current_guess,
            terminal_mode = terminal_mode,
            output_N = max(output_N, Int(clamp(round(stage_T) + 1, p.N, 401))))
        last_stage_result = stage_result
        if abs(stage_T - p.T) <= 1e-12
            target_result = stage_result
        end
        progress && print_progress_result("shooting stage", stage_result)
    end
    return target_result === nothing ? last_stage_result : target_result
end

"""
    multiple_shooting_segment_times(p; segment_length=2.0)

Builds the segment boundary times used by the multiple-shooting solver.

Input arguments:
- `p::ModelParams`: model parameters.

Optional parameters:
- `segment_length::Real = 2.0`: target maximum segment length.

Output:
- Returns a `Vector{Float64}` containing `m+1` boundary times from `0.0` to `p.T`.
"""
function multiple_shooting_segment_times(p::ModelParams; segment_length::Real = 2.0)
    max_length = max(Float64(segment_length), 1e-6)
    segments = max(1, Int(ceil(p.T / max_length)))
    return collect(range(0.0, p.T, length = segments + 1))
end

"""
    shooting_path_guess(p, steady, tgrid; reference=nothing)

Builds a path guess on an arbitrary time grid for shooting-based solvers.

Input arguments:
- `p::ModelParams`: model parameters.
- `steady::SteadyStateResult`: steady-state reference.
- `tgrid::AbstractVector`: target time grid.

Optional parameters:
- `reference=nothing`: optional reference path used as an interpolated seed.

Output:
- Returns a flat vector of length `6N`, where `N = length(tgrid)`.
"""
function shooting_path_guess(p::ModelParams, steady::SteadyStateResult, tgrid::AbstractVector; reference = nothing)
    if reference === nothing
        return collocation_guess(p, steady, tgrid)
    end

    reference_t = reference.t[end] <= 0 ? reference.t : rescale_time_grid(reference.t, tgrid[end])
    return interpolate_guess(reference_t, pack_solution(reference), tgrid)
end

"""
    multiple_shooting_unknowns_from_guess(boundary_guess, segment_count, p)

Packs boundary-state guesses into the unknown vector used by multiple shooting.

Input arguments:
- `boundary_guess::AbstractVector`: flat vector of length `6(segment_count+1)` on the segment boundary grid.
- `segment_count::Int`: number of shooting segments.
- `p::ModelParams`: model parameters.

Optional parameters:
- None.

Output:
- Returns a `Vector{Float64}` of length `3 + 6(segment_count-1)`.
"""
function multiple_shooting_unknowns_from_guess(boundary_guess::AbstractVector, segment_count::Int, p::ModelParams)
    guess0 = node_slice(boundary_guess, 1)
    unknowns = Float64[
        log(max(guess0[2], p.min_positive)),
        guess0[4],
        guess0[6],
    ]
    for segment_idx in 2:segment_count
        node_guess = node_slice(boundary_guess, segment_idx)
        append!(unknowns, Float64[
            node_guess[1],
            log(max(node_guess[2], p.min_positive)),
            node_guess[3],
            node_guess[4],
            node_guess[5],
            node_guess[6],
        ])
    end
    return unknowns
end

"""
    unpack_multiple_shooting_unknowns(v, p, segment_count)

Reconstructs the segment start states from the multiple-shooting unknown vector.

Input arguments:
- `v::AbstractVector`: vector of length `3 + 6(segment_count-1)`.
- `p::ModelParams`: model parameters.
- `segment_count::Int`: number of shooting segments.

Optional parameters:
- None.

Output:
- Returns a vector of `segment_count` state vectors, each of length 6.
"""
function unpack_multiple_shooting_unknowns(v::AbstractVector, p::ModelParams, segment_count::Int)
    starts = Vector{Vector{Float64}}(undef, segment_count)
    starts[1] = [p.k0, exp(v[1]), p.q0, v[2], p.Λ20, v[3]]
    cursor = 4
    for segment_idx in 2:segment_count
        starts[segment_idx] = [
            v[cursor],
            exp(v[cursor + 1]),
            v[cursor + 2],
            v[cursor + 3],
            v[cursor + 4],
            v[cursor + 5],
        ]
        cursor += 6
    end
    return starts
end

"""
    multiple_shooting_residual!(residual, v, p, steady, segment_times; terminal_mode=:tvc)

Assembles the multiple-shooting continuity and terminal residuals in place.

Input arguments:
- `residual`: residual vector of length `3 + 6(segment_count-1)`.
- `v::AbstractVector`: unknown vector of length `3 + 6(segment_count-1)`.
- `p::ModelParams`: model parameters.
- `steady::SteadyStateResult`: steady-state reference.
- `segment_times::AbstractVector`: segment boundary times of length `segment_count+1`.

Optional parameters:
- `terminal_mode::Symbol = :tvc`: terminal closure rule.

Output:
- Returns `nothing`.
"""
function multiple_shooting_residual!(residual, v::AbstractVector, p::ModelParams, steady::SteadyStateResult, segment_times::AbstractVector; terminal_mode::Symbol = :tvc)
    segment_count = length(segment_times) - 1
    scales = shooting_scales(p, steady)
    starts = unpack_multiple_shooting_unknowns(v, p, segment_count)
    cursor = 1

    for segment_idx in 1:segment_count
        start_state = starts[segment_idx]
        if !state_path_is_admissible(start_state, p)
            fill!(residual, 1e6)
            return nothing
        end

        local_horizon = segment_times[segment_idx + 1] - segment_times[segment_idx]
        local_params = with_horizon(p, local_horizon, p.N)
        sol = solve_shooting_ivp(local_params, start_state)
        if sol === nothing || sol.retcode != SciMLBase.ReturnCode.Success || isempty(sol.u)
            fill!(residual, 1e6)
            return nothing
        end

        yend = sol.u[end]
        if !state_path_is_admissible(yend, p)
            fill!(residual, 1e6)
            return nothing
        end

        if segment_idx < segment_count
            residual[cursor:cursor + 5] .= (yend .- starts[segment_idx + 1]) ./ collect(scales)
            cursor += 6
        else
            residual[cursor:cursor + 2] .= terminal_residual_values(yend, p, steady, scales, p.T, terminal_mode)
        end
    end
    return nothing
end

"""
    multiple_shooting_result(v, p, steady, segment_times; output_N=max(p.N, 201), terminal_mode=:tvc)

Projects a multiple-shooting candidate to a structured path on a common output grid.

Input arguments:
- `v::AbstractVector`: multiple-shooting unknown vector.
- `p::ModelParams`: model parameters.
- `steady::SteadyStateResult`: steady-state reference.
- `segment_times::AbstractVector`: segment boundary times.

Optional parameters:
- `output_N::Int = max(p.N, 201)`: output grid size.
- `terminal_mode::Symbol = :tvc`: terminal closure rule.

Output:
- Returns a `CollocationResult`.
"""
function multiple_shooting_result(v::AbstractVector, p::ModelParams, steady::SteadyStateResult, segment_times::AbstractVector;
    output_N::Int = max(p.N, 201),
    terminal_mode::Symbol = :tvc)
    segment_count = length(segment_times) - 1
    starts = unpack_multiple_shooting_unknowns(v, p, segment_count)
    output_t = collect(range(0.0, p.T, length = output_N))
    z = zeros(6 * output_N)
    output_index = 1

    for segment_idx in 1:segment_count
        start_time = segment_times[segment_idx]
        end_time = segment_times[segment_idx + 1]
        local_horizon = end_time - start_time
        local_params = with_horizon(p, local_horizon, p.N)
        local_indices = segment_idx == 1 ?
            findall(t -> start_time - 1e-12 <= t <= end_time + 1e-12, output_t) :
            findall(t -> start_time + 1e-12 < t <= end_time + 1e-12, output_t)
        local_saveat = output_t[local_indices] .- start_time
        sol = solve_shooting_ivp(local_params, starts[segment_idx]; saveat = local_saveat)
        if sol === nothing || sol.retcode != SciMLBase.ReturnCode.Success || length(sol.u) != length(local_indices)
            residual = zeros(length(v))
            multiple_shooting_residual!(residual, v, p, steady, segment_times; terminal_mode = terminal_mode)
            return unpack_solution(z, p, steady, output_t, maximum(abs.(residual)), false)
        end

        for (local_pos, global_pos) in enumerate(local_indices)
            z[node_offset(global_pos) + 1:node_offset(global_pos) + 6] .= sol.u[local_pos]
            output_index = global_pos
        end
    end

    residual = zeros(length(v))
    multiple_shooting_residual!(residual, v, p, steady, segment_times; terminal_mode = terminal_mode)
    resnorm = maximum(abs.(residual))
    success = isfinite(resnorm) && resnorm <= p.residual_tolerance
    return unpack_solution(z, p, steady, output_t, resnorm, success)
end

"""
    solve_multiple_shooting_stage(p, steady; progress=true, terminal_mode=:tvc, reference=nothing, segment_length=2.0, output_N=max(p.N, 201))

Solves one horizon stage with multiple shooting.

Input arguments:
- `p::ModelParams`: model parameters.
- `steady::SteadyStateResult`: steady-state reference.

Optional parameters:
- `progress::Bool = true`: print progress messages.
- `terminal_mode::Symbol = :tvc`: terminal closure rule.
- `reference=nothing`: optional reference path used to seed the segment boundary states.
- `segment_length::Real = 2.0`: target maximum segment length.
- `output_N::Int = max(p.N, 201)`: output grid size for the returned path.

Output:
- Returns a `CollocationResult`.
"""
function solve_multiple_shooting_stage(p::ModelParams, steady::SteadyStateResult;
    progress::Bool = true,
    terminal_mode::Symbol = :tvc,
    reference = nothing,
    segment_length::Real = 2.0,
    output_N::Int = max(p.N, 201))
    best_result = nothing
    segment_lengths = sort(unique([Float64(segment_length), 1.0]), rev = true)
    for active_segment_length in segment_lengths
        segment_times = multiple_shooting_segment_times(p; segment_length = active_segment_length)
        guesses = Vector{Vector{Float64}}()
        if reference !== nothing
            push!(guesses, shooting_path_guess(p, steady, segment_times; reference = reference))
        end
        push!(guesses, shooting_path_guess(p, steady, segment_times; reference = nothing))

        for (guess_idx, boundary_guess) in enumerate(guesses)
            v0 = multiple_shooting_unknowns_from_guess(boundary_guess, length(segment_times) - 1, p)
            residual!(F, v) = multiple_shooting_residual!(F, v, p, steady, segment_times; terminal_mode = terminal_mode)
            solver = try
                nlsolve(residual!, v0;
                    method = :trust_region,
                    iterations = p.max_iter,
                    xtol = 1e-10,
                    ftol = 1e-10,
                    show_trace = false)
            catch
                nothing
            end

            result = solver === nothing ?
                multiple_shooting_result(v0, p, steady, segment_times; output_N = output_N, terminal_mode = terminal_mode) :
                multiple_shooting_result(solver.zero, p, steady, segment_times; output_N = output_N, terminal_mode = terminal_mode)
            best_result = best_result === nothing ? result : better_target_result(result, best_result)
            if progress
                print_progress_result("multiple shooting L=$(active_segment_length) candidate #$(guess_idx)", result)
            end
        end
    end

    progress && print_progress_result("multiple shooting stage", best_result)
    return best_result
end

"""
    solve_multiple_shooting(p=ModelParams(); progress=true, terminal_mode=:tvc, use_horizon_continuation=true, output_N=max(p.N, 201), segment_length=2.0)

High-level multiple-shooting solver for the `OptimalWealthTax` model.

Input arguments:
- `p::ModelParams = ModelParams()`: model parameters.

Optional parameters:
- `progress::Bool = true`: print progress messages.
- `terminal_mode::Symbol = :tvc`: terminal closure rule.
- `use_horizon_continuation::Bool = true`: solve intermediate horizons before the target horizon.
- `output_N::Int = max(p.N, 201)`: output grid size for the returned path.
- `segment_length::Real = 2.0`: target maximum segment length.

Output:
- Returns a `CollocationResult` from the target horizon stage.
"""
function solve_multiple_shooting(p::ModelParams = ModelParams();
    progress::Bool = true,
    terminal_mode::Symbol = :tvc,
    use_horizon_continuation::Bool = true,
    output_N::Int = max(p.N, 201),
    segment_length::Real = 2.0)
    steady = find_steady_state(p)
    T_stages = use_horizon_continuation ? default_shooting_horizon_stages(p.T) : [p.T]
    reference = nothing
    target_result = nothing
    last_stage_result = nothing

    progress && println("OptimalWealthTax multiple shooting start: terminal_mode=$(terminal_mode) T=$(p.T) k0=$(p.k0) q0=$(p.q0) Λ20=$(p.Λ20)")
    for stage_T in T_stages
        stage_params = with_horizon(p, stage_T, p.N)
        progress && println("OptimalWealthTax multiple shooting stage T=$(stage_T)")
        stage_result = solve_multiple_shooting_stage(stage_params, steady;
            progress = progress,
            terminal_mode = terminal_mode,
            reference = reference,
            segment_length = segment_length,
            output_N = max(output_N, Int(clamp(round(stage_T) + 1, p.N, 401))))
        reference = stage_result
        last_stage_result = stage_result
        if abs(stage_T - p.T) <= 1e-12
            target_result = stage_result
        end
    end
    return target_result === nothing ? last_stage_result : target_result
end

"""
    solve_collocation_problem(p, steady; N=p.N, progress=true, initial_t=nothing, initial_z=nothing, use_mesh_continuation=true, terminal_mode=:steady_state, terminal_alpha=1.0)

Solves one collocation problem, optionally using a sequence of coarser meshes before the final grid.

Input arguments:
- `p::ModelParams`: model parameters.
- `steady::SteadyStateResult`: steady-state reference.

Optional parameters:
- `N::Int = p.N`: target number of collocation nodes.
- `progress::Bool = true`: print stage-level progress.
- `initial_t`: optional previous time grid of length `N_prev`.
- `initial_z`: optional previous flat solution vector of length `6N_prev`.
- `use_mesh_continuation::Bool = true`: whether to solve on intermediate mesh sizes first.
- `terminal_mode::Symbol = :steady_state`: terminal closure rule.
- `terminal_alpha::Float64 = 1.0`: homotopy weight for TVC continuation.

Output:
- Returns a tuple `(final_result, previous_t, previous_z)`.
- `final_result` is a `CollocationResult` with trajectory length equal to the last mesh size used.
- `previous_t` has length `N_last`, and `previous_z` has length `6N_last`.
"""
function solve_collocation_problem(p::ModelParams, steady::SteadyStateResult; N::Int = p.N, progress::Bool = true, initial_t = nothing, initial_z = nothing, use_mesh_continuation::Bool = true, terminal_mode::Symbol = :steady_state, terminal_alpha::Float64 = 1.0)
    terminal_mode = canonical_terminal_mode(terminal_mode)
    stage_sizes = use_mesh_continuation ? unique(max.(11, [cld(N, 3), cld(2 * N, 3), N])) : [N]

    previous_t = initial_t
    previous_z = initial_z
    final_result = nothing

    for Ncur in stage_sizes
        tgrid = collocation_grid(p, Ncur)
        guess = previous_z === nothing ? collocation_guess(p, steady, tgrid) : interpolate_guess(previous_t, previous_z, tgrid)
        residual!(F, z) = collocation_residual!(F, z, p, steady, tgrid; terminal_mode = terminal_mode, terminal_alpha = terminal_alpha)
        progress && println("OptimalWealthTax collocation stage N=$(Ncur)")
        F = zeros(length(guess))
        residual!(F, guess)
        progress && println("  initial residual=$(maximum(abs.(F)))")
        nls = solve_nonlinear_system(residual!, guess, p; show_trace = progress)
        z = nls === nothing ? guess : nls.zero
        residual!(F, z)
        resnorm = maximum(abs.(F))
        success = isfinite(resnorm) && resnorm <= p.residual_tolerance
        previous_t = collect(tgrid)
        previous_z = copy(z)
        final_result = unpack_solution(previous_z, p, steady, previous_t, resnorm, success)
        progress && print_progress_result("stage N=$(Ncur)", final_result)

        if p.active_set_iterations > 0 && !final_result.success && should_try_active_set_stage(final_result, p)
            progress && println("OptimalWealthTax collocation active-set fallback N=$(Ncur)")
            active_result, active_z = solve_active_set_stage(p, steady, tgrid, previous_z;
                progress = progress,
                terminal_mode = terminal_mode,
                terminal_alpha = terminal_alpha,
                max_active_set_iterations = p.active_set_iterations)
            if better_target_result(active_result, final_result) === active_result
                final_result = active_result
                previous_z = active_z
            end
            progress && print_progress_result("stage N=$(Ncur) active-set", final_result)
        end

        if !success
            break
        end
    end

    return final_result, previous_t, previous_z
end

"""
    continue_horizon(p, steady; N=p.N, progress=true, target_T=p.T, terminal_mode=:steady_state, terminal_alpha=1.0)

Performs horizon continuation from the default short horizon to a target horizon.

Input arguments:
- `p::ModelParams`: target parameter set.
- `steady::SteadyStateResult`: steady-state reference.

Optional parameters:
- `N::Int = p.N`: target number of nodes at the final horizon.
- `progress::Bool = true`: print continuation progress.
- `target_T::Real = p.T`: target final horizon.
- `terminal_mode::Symbol = :steady_state`: terminal closure rule.
- `terminal_alpha::Float64 = 1.0`: homotopy weight for TVC continuation.

Output:
- Returns `(final_result, previous_t, previous_z)`.
- The result trajectories have length equal to the last stage mesh size.
- `previous_z` is a flat vector of length `6N_last`.
"""
function continue_horizon(p::ModelParams, steady::SteadyStateResult; N::Int = p.N, progress::Bool = true, target_T::Real = p.T, terminal_mode::Symbol = :steady_state, terminal_alpha::Float64 = 1.0)
    base_T = ModelParams().T
    if target_T <= base_T + 1e-12
        return solve_collocation_problem(p, steady; N = N, progress = progress, use_mesh_continuation = true, terminal_mode = terminal_mode, terminal_alpha = terminal_alpha)
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
            use_mesh_continuation = previous_z === nothing,
            terminal_mode = terminal_mode,
            terminal_alpha = terminal_alpha)
        progress && print_progress_result("horizon T=$(round(T_stage; digits = 4))", final_result)
        if !final_result.success
            break
        end
    end

    return final_result, previous_t, previous_z
end

"""
    continue_horizon_stages(p, steady, T_stages; N=p.N, progress=true, terminal_mode=:steady_state, terminal_alpha=1.0)

Performs horizon continuation on an explicit user-provided list of horizon values.

Input arguments:
- `p::ModelParams`: target parameter set.
- `steady::SteadyStateResult`: steady-state reference.
- `T_stages::AbstractVector{<:Real}`: ordered vector of scalar horizons.

Optional parameters:
- `N::Int = p.N`: target number of nodes at the last stage.
- `progress::Bool = true`: print stage-level progress.
- `terminal_mode::Symbol = :steady_state`: terminal closure rule.
- `terminal_alpha::Float64 = 1.0`: homotopy weight for TVC continuation.

Output:
- Returns `(final_result, previous_t, previous_z)`.
- `final_result` stores trajectories on the last successful stage.
- `previous_t` and `previous_z` describe the same stage in grid and flat-vector form.
"""
function continue_horizon_stages(p::ModelParams, steady::SteadyStateResult, T_stages::AbstractVector{<:Real};
    N::Int = p.N,
    progress::Bool = true,
    terminal_mode::Symbol = :steady_state,
    terminal_alpha::Float64 = 1.0,
    acceptance_tolerance = nothing)
    isempty(T_stages) && error("T_stages must contain at least one horizon value")

    previous_t = nothing
    previous_z = nothing
    final_result = nothing
    target_T = Float64(last(T_stages))

    for (stage_idx, T_stage_raw) in enumerate(T_stages)
        T_stage = Float64(T_stage_raw)
        N_stage = max(11, round(Int, 1 + (N - 1) * T_stage / target_T))
        stage_params = with_horizon(p, T_stage, N_stage)
        progress && println("OptimalWealthTax staged horizon continuation T=$(round(T_stage; digits = 4)) N=$(N_stage)")

        if stage_idx == 1
            final_result = solve_collocation(stage_params;
                N = N_stage,
                progress = progress,
                use_continuation = true,
                use_bvp_refinement = false,
                use_horizon_continuation = false,
                terminal_mode = terminal_mode,
                use_nested_seed = is_tvc_like_terminal_mode(terminal_mode))
            previous_t = final_result.t
            previous_z = pack_solution(final_result)
        else
            stage_t = rescale_time_grid(previous_t, T_stage)
            propagated_result, propagated_t, propagated_z = solve_collocation_problem(stage_params, steady;
                N = N_stage,
                progress = progress,
                initial_t = stage_t,
                initial_z = previous_z,
                use_mesh_continuation = true,
                terminal_mode = terminal_mode,
                terminal_alpha = terminal_alpha)
            final_result = propagated_result
            previous_t = propagated_t
            previous_z = propagated_z

            accepted_stage = final_result.success ||
                (acceptance_tolerance !== nothing && isfinite(final_result.residual_norm) && final_result.residual_norm <= acceptance_tolerance)

            if !accepted_stage
                progress && println("OptimalWealthTax staged horizon fallback solve at T=$(round(T_stage; digits = 4))")
                fallback_result = solve_collocation(stage_params;
                    N = N_stage,
                    progress = progress,
                    use_continuation = true,
                    use_bvp_refinement = false,
                    use_horizon_continuation = false,
                    terminal_mode = terminal_mode,
                    use_nested_seed = is_tvc_like_terminal_mode(terminal_mode))
                final_result = better_target_result(propagated_result, fallback_result)
                previous_t = final_result.t
                previous_z = pack_solution(final_result)
            end
        end

        progress && print_progress_result("staged horizon T=$(round(T_stage; digits = 4))", final_result)
        accepted_stage = final_result.success ||
            (acceptance_tolerance !== nothing && isfinite(final_result.residual_norm) && final_result.residual_norm <= acceptance_tolerance)
        if !accepted_stage
            break
        end
    end

    return final_result, previous_t, previous_z
end

function default_horizon_stages(target_T::Real)
    target = Float64(target_T)
    stages = Float64[]
    for T_stage in (10.0, 20.0, 40.0, 80.0, 120.0, 160.0, target)
        if T_stage <= target + 1e-12
            push!(stages, T_stage)
        end
    end
    isempty(stages) && push!(stages, target)
    return unique(sort(stages))
end

horizon_reached_target(result::CollocationResult, target_T::Real) = !isempty(result.t) && abs(result.t[end] - Float64(target_T)) <= 1e-8

"""
    solve_collocation_staged_horizon(p, T_stages; N=p.N, progress=true, terminal_mode=:state_steady_state)

Convenience wrapper that computes the steady state and then runs explicit staged horizon continuation.

Input arguments:
- `p::ModelParams`: target parameter set.
- `T_stages::AbstractVector{<:Real}`: ordered horizon stages.

Optional parameters:
- `N::Int = p.N`: target node count for the final stage.
- `progress::Bool = true`: print progress.
- `terminal_mode::Symbol = :state_steady_state`: terminal closure rule.

Output:
- Returns `(final_result, previous_t, previous_z)`.
- `final_result` is a `CollocationResult`; `previous_z` is a flat vector of length `6N_last`.
"""
function solve_collocation_staged_horizon(p::ModelParams, T_stages::AbstractVector{<:Real};
    N::Int = p.N,
    progress::Bool = true,
    terminal_mode::Symbol = :state_steady_state)
    steady = find_steady_state(p)
    progress && println("OptimalWealthTax staged solve start: terminal_mode=$(terminal_mode) target T=$(p.T) stages=$(collect(Float64.(T_stages)))")
    return continue_horizon_stages(p, steady, T_stages;
        N = N,
        progress = progress,
        terminal_mode = terminal_mode)
end

"""
    solve_bvp_problem(p, steady; N=p.N, progress=true, initial_t, initial_z, terminal_mode=:steady_state)

Runs the BVP refinement step using `BoundaryValueDiffEq` and projects the result back onto the collocation grid.

Input arguments:
- `p::ModelParams`: model parameters.
- `steady::SteadyStateResult`: steady-state reference.

Optional parameters:
- `N::Int = p.N`: number of output grid nodes.
- `progress::Bool = true`: print progress messages.
- `initial_t`: initial guess time grid of length `N_init`.
- `initial_z`: initial flat guess vector of length `6N_init`.
- `terminal_mode::Symbol = :steady_state`: terminal closure rule.

Output:
- Returns a `CollocationResult` with trajectory length `N`.
- If the BVP solver fails, the returned result is built from the provided seed after interpolation.
"""
function solve_bvp_problem(p::ModelParams, steady::SteadyStateResult; N::Int = p.N, progress::Bool = true, initial_t, initial_z, terminal_mode::Symbol = :steady_state)
    terminal_mode = canonical_terminal_mode(terminal_mode)
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
        res[3] = ua[5] - p.Λ20
        if terminal_mode == :steady_state || terminal_mode == :costate_steady_state
            res[4] = ub[2] - steady.c
            res[5] = ub[4] - steady.Λ1
            res[6] = ub[6] - steady.Λ3
        elseif terminal_mode == :state_steady_state
            res[4] = ub[1] - steady.k
            res[5] = ub[3] - steady.q
            res[6] = ub[2] - steady.c
        elseif terminal_mode == :state_stationary
            controls = foc_implied_controls(ub, p)
            if controls === nothing
                res[4] = 1e6
                res[5] = 1e6
                res[6] = 1e6
            else
                res[4] = ub[1] - steady.k
                res[5] = controls.r_tilde - p.ρ
                res[6] = ub[3] - steady.q
            end
        elseif is_tvc_like_terminal_mode(terminal_mode)
            discount = exp(-p.ρ * p.T)
            res[4] = discount * ub[4] * ub[1]
            res[5] = discount * terminal_middle_metric(ub[2], ub[1], ub[5], p, terminal_mode)
            res[6] = discount * ub[6] * ub[3]
        else
            error("Unsupported terminal_mode=$(terminal_mode)")
        end
        return nothing
    end

    function guess_y(t)
        if initial_t === nothing || initial_z === nothing
            w = p.T <= 0 ? 0.0 : t / p.T
            k_guess = (1.0 - w) * p.k0 + w * steady.k
            q_guess = (1.0 - w) * p.q0 + w * steady.q
            Λ2_guess = (1.0 - w) * p.Λ20 + w * steady.Λ2
            c_guess, Λ1_guess, Λ3_guess = collocation_guess_values(k_guess, q_guess, Λ2_guess, steady, p)
            return [
                k_guess,
                c_guess,
                q_guess,
                Λ1_guess,
                Λ2_guess,
                Λ3_guess,
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
    collocation_residual!(residual, z, p, steady, tgrid; terminal_mode = terminal_mode)
    resnorm = maximum(abs.(residual))
    success = SciMLBase.successful_retcode(sol.retcode) && isfinite(resnorm) && resnorm <= p.residual_tolerance
    return unpack_solution(z, p, steady, tgrid, resnorm, success)
end

"""
    compare_solution_paths(reference, candidate)

Compares two collocation solutions on a common grid and reports normalized path differences.

Input arguments:
- `reference::CollocationResult`: baseline solution with trajectory length `N_ref`.
- `candidate::CollocationResult`: solution to compare; it may live on a different grid.

Optional parameters:
- None.

Output:
- Returns a named tuple of scalar diagnostics and nested named tuples.
- The output includes scalar success flags, residuals, terminal changes, and maximum normalized component-wise deviations.
"""
function compare_solution_paths(reference::CollocationResult, candidate::CollocationResult)
    candidate_z = length(reference.t) == length(candidate.t) && all(reference.t .== candidate.t) ?
        pack_solution(candidate) :
        interpolate_guess(candidate.t, pack_solution(candidate), reference.t)

    scales = (
        max(maximum(abs.(reference.k)), maximum(abs.(candidate.k)), 1.0),
        max(maximum(abs.(reference.c)), maximum(abs.(candidate.c)), 1.0),
        max(maximum(abs.(reference.q)), maximum(abs.(candidate.q)), 1.0),
        max(maximum(abs.(reference.Λ1)), maximum(abs.(candidate.Λ1)), 1.0),
        max(maximum(abs.(reference.Λ2)), maximum(abs.(candidate.Λ2)), 1.0),
        max(maximum(abs.(reference.Λ3)), maximum(abs.(candidate.Λ3)), 1.0),
    )

    max_change = (
        k = maximum(abs.(reference.k .- [candidate_z[node_offset(i) + 1] for i in eachindex(reference.t)])) / scales[1],
        c = maximum(abs.(reference.c .- [candidate_z[node_offset(i) + 2] for i in eachindex(reference.t)])) / scales[2],
        q = maximum(abs.(reference.q .- [candidate_z[node_offset(i) + 3] for i in eachindex(reference.t)])) / scales[3],
        Λ1 = maximum(abs.(reference.Λ1 .- [candidate_z[node_offset(i) + 4] for i in eachindex(reference.t)])) / scales[4],
        Λ2 = maximum(abs.(reference.Λ2 .- [candidate_z[node_offset(i) + 5] for i in eachindex(reference.t)])) / scales[5],
        Λ3 = maximum(abs.(reference.Λ3 .- [candidate_z[node_offset(i) + 6] for i in eachindex(reference.t)])) / scales[6],
    )

    return (
        success_before = reference.success,
        success_after = candidate.success,
        residual_before = reference.residual_norm,
        residual_after = candidate.residual_norm,
        residual_ratio = candidate.residual_norm / max(reference.residual_norm, eps()),
        max_normalized_change = max_change,
        terminal_change = (
            k = candidate.k[end] - reference.k[end],
            c = candidate.c[end] - reference.c[end],
            q = candidate.q[end] - reference.q[end],
            Λ1 = candidate.Λ1[end] - reference.Λ1[end],
            Λ2 = candidate.Λ2[end] - reference.Λ2[end],
            Λ3 = candidate.Λ3[end] - reference.Λ3[end],
        ),
    )
end

"""
    refine_with_bvp(reference, p; N=length(reference.t), progress=true, terminal_mode=:state_steady_state)

Runs BVP refinement starting from an existing collocation solution and returns both the refined path and verification diagnostics.

Input arguments:
- `reference::CollocationResult`: seed solution with trajectory length `N_ref`.
- `p::ModelParams`: model parameters.

Optional parameters:
- `N::Int = length(reference.t)`: output grid size for the refined solution.
- `progress::Bool = true`: print refinement diagnostics.
- `terminal_mode::Symbol = :state_steady_state`: terminal closure rule.

Output:
- Returns a named tuple with fields `reference`, `refined`, `verification`, and `preserved_reference`.
- `reference` and `refined` are `CollocationResult` objects, each with trajectory length `N` after refinement output is formed.
"""
function refine_with_bvp(reference::CollocationResult, p::ModelParams;
    N::Int = length(reference.t),
    progress::Bool = true,
    terminal_mode::Symbol = :state_steady_state)
    refined = solve_bvp_problem(p, reference.steady;
        N = N,
        progress = progress,
        initial_t = reference.t,
        initial_z = pack_solution(reference),
        terminal_mode = terminal_mode)
    verification = compare_solution_paths(reference, refined)
    max_change = maximum(values(verification.max_normalized_change))
    preserved_reference = max_change <= 1e-12

    if progress
        println("OptimalWealthTax BVP verification")
        println("  residual before=$(verification.residual_before) after=$(verification.residual_after)")
        println("  success  before=$(verification.success_before) after=$(verification.success_after)")
        println("  max normalized change=$(max_change)")
        println("  preserved reference path=$(preserved_reference)")
    end

    return (; reference, refined, verification, preserved_reference)
end

"""
    continue_initial_conditions(p, steady, previous_t, previous_z; N=p.N, progress=true, base_step=0.025, min_step=1e-4, terminal_mode=:steady_state, terminal_alpha=1.0, label, endpoint)

Performs continuation in the initial conditions starting from a previously solved branch.

Input arguments:
- `p::ModelParams`: target parameter set.
- `steady::SteadyStateResult`: steady-state reference.
- `previous_t`: current time grid of length `N_prev`.
- `previous_z`: current flat solution vector of length `6N_prev`.

Optional parameters:
- `N::Int = p.N`: node count used in each continuation attempt.
- `progress::Bool = true`: print progress.
- `base_step::Float64 = 0.025`: initial continuation step size in homotopy parameter space.
- `min_step::Float64 = 1e-4`: minimum accepted continuation step size.
- `terminal_mode::Symbol = :steady_state`: terminal closure rule.
- `terminal_alpha::Float64 = 1.0`: homotopy weight for TVC continuation.
- `label::AbstractString`: label printed during continuation.
- `endpoint`: callable mapping a scalar `α` to either `(k0, q0)` or `(k0, q0, Λ20)`.

Output:
- Returns `(current_alpha, final_result, previous_t, previous_z)`.
- `current_alpha` is a scalar in `[0, 1]`.
- `final_result` is the last accepted `CollocationResult`, while `previous_z` has length `6N_last`.
"""
function continue_initial_conditions(p::ModelParams, steady::SteadyStateResult, previous_t, previous_z;
    N::Int = p.N,
    progress::Bool = true,
    base_step::Float64 = 0.005,
    min_step::Float64 = 1e-5,
    acceptance_tolerance = nothing,
    terminal_mode::Symbol = :steady_state,
    terminal_alpha::Float64 = 1.0,
    label::AbstractString,
    endpoint)
    current_alpha = 0.0
    current_step = base_step
    previous_alpha = nothing
    previous_previous_t = nothing
    previous_previous_z = nothing
    final_result = unpack_solution(previous_z, p, steady, previous_t, 0.0, false)

    while current_alpha < 1.0 - 1e-12
        step = min(current_step, 1.0 - current_alpha)
        step_success = false

        while step >= min_step - 1e-12
            next_alpha = min(1.0, current_alpha + step)
            target = endpoint(next_alpha)
            if length(target) == 2
                k0, q0 = target
                Λ20 = p.Λ20
            elseif length(target) == 3
                k0, q0, Λ20 = target
            else
                error("Continuation endpoint must return (k0, q0) or (k0, q0, Λ20)")
            end
            trial_params = with_initial_conditions(p, k0, q0; Λ20 = Λ20)

            trial_initial_z = previous_z
            if previous_alpha !== nothing && previous_previous_t !== nothing && previous_previous_z !== nothing &&
               length(previous_previous_t) == length(previous_t) && length(previous_previous_z) == length(previous_z) &&
               all(previous_previous_t .== previous_t)
                denom_alpha = current_alpha - previous_alpha
                if abs(denom_alpha) > 1e-12
                    predictor_weight = (next_alpha - current_alpha) / denom_alpha
                    if isfinite(predictor_weight) && predictor_weight > 0.0
                        predicted_z = previous_z .+ predictor_weight .* (previous_z .- previous_previous_z)
                        if all(isfinite, predicted_z)
                            trial_initial_z = predicted_z
                        end
                    end
                end
            end

            progress && println("OptimalWealthTax $(label) continuation α=$(round(next_alpha; digits = 4))")
            trial_result, trial_t, trial_z = solve_collocation_problem(trial_params, steady;
                N = N,
                progress = progress,
                initial_t = previous_t,
                initial_z = trial_initial_z,
                use_mesh_continuation = false,
                terminal_mode = terminal_mode,
                terminal_alpha = terminal_alpha)

            accepted = trial_result.success ||
                (acceptance_tolerance !== nothing && isfinite(trial_result.residual_norm) && trial_result.residual_norm <= acceptance_tolerance)

            if accepted
                previous_alpha = current_alpha
                previous_previous_t = copy(previous_t)
                previous_previous_z = copy(previous_z)
                current_alpha = next_alpha
                current_step = min(base_step, max(step, min_step))
                previous_t = trial_t
                previous_z = trial_z
                final_result = trial_result
                progress && print_progress_result("$(label) continuation accepted α=$(round(current_alpha; digits = 4))", final_result)
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

"""
    continue_terminal_conditions(p, steady, previous_t, previous_z; N=p.N, progress=true, base_step=0.1, min_step=1e-5, acceptance_tolerance=max(1e-4, 10.0 * p.residual_tolerance))

Performs terminal-condition homotopy from state-steady-state closure toward full TVC closure.

Input arguments:
- `p::ModelParams`: model parameters.
- `steady::SteadyStateResult`: steady-state reference.
- `previous_t`: current time grid of length `N_prev`.
- `previous_z`: current flat solution vector of length `6N_prev`.

Optional parameters:
- `N::Int = p.N`: node count for each solve.
- `progress::Bool = true`: print progress.
- `base_step::Float64 = 0.1`: initial homotopy step size.
- `min_step::Float64 = 1e-5`: minimum homotopy step size.
- `acceptance_tolerance`: scalar threshold used to accept intermediate non-final TVC solves.

Output:
- Returns `(current_alpha, final_result, previous_t, previous_z)`.
- `current_alpha` is the accepted TVC homotopy level in `[0, 1]`.
- `final_result` is the last accepted `CollocationResult`.
"""
function continue_terminal_conditions(p::ModelParams, steady::SteadyStateResult, previous_t, previous_z;
    N::Int = p.N,
    progress::Bool = true,
    base_step::Float64 = 0.1,
    min_step::Float64 = 1e-5,
    acceptance_tolerance::Float64 = max(1e-4, 10.0 * p.residual_tolerance),
    terminal_mode::Symbol = :tvc)
    terminal_mode = canonical_terminal_mode(terminal_mode)
    initial_terminal_mode = if is_tvc_like_terminal_mode(terminal_mode)
        :state_steady_state
    elseif terminal_mode == :state_steady_state
        :costate_steady_state
    else
        error("Unsupported terminal continuation terminal_mode=$(terminal_mode)")
    end
    continuation_label = is_tvc_like_terminal_mode(terminal_mode) ? "terminal continuation" : "state-terminal continuation"
    current_alpha = 0.0
    final_result = evaluate_candidate(previous_z, p, steady, previous_t;
        success = false,
        terminal_mode = current_alpha <= 1e-12 ? initial_terminal_mode : terminal_mode,
        terminal_alpha = current_alpha)
    best_trial_alpha = current_alpha
    best_trial_result = final_result
    best_trial_t = previous_t
    best_trial_z = previous_z

    while current_alpha < 1.0 - 1e-12
        step = min(base_step, 1.0 - current_alpha)
        step_success = false

        while step >= min_step - 1e-12
            next_alpha = min(1.0, current_alpha + step)
            progress && println("OptimalWealthTax $(continuation_label) α=$(round(next_alpha; digits = 4))")
            trial_result, trial_t, trial_z = solve_collocation_problem(p, steady;
                N = N,
                progress = progress,
                initial_t = previous_t,
                initial_z = previous_z,
                use_mesh_continuation = false,
                terminal_mode = terminal_mode,
                terminal_alpha = next_alpha)

            if isfinite(trial_result.residual_norm)
                best_is_initial = best_trial_alpha <= 1e-12
                better_trial = !isfinite(best_trial_result.residual_norm) || trial_result.success ||
                    (best_is_initial && next_alpha > 1e-12) ||
                    (trial_result.residual_norm < best_trial_result.residual_norm)
                if better_trial
                    best_trial_alpha = next_alpha
                    best_trial_result = trial_result
                    best_trial_t = trial_t
                    best_trial_z = trial_z
                end
            end

            if isfinite(trial_result.residual_norm) && trial_result.residual_norm <= acceptance_tolerance
                current_alpha = next_alpha
                previous_t = trial_t
                previous_z = trial_z
                final_result = trial_result
                progress && print_progress_result("$(continuation_label) accepted α=$(round(current_alpha; digits = 4))", final_result)
                step_success = true
                break
            end

            step *= 0.5
            if step >= min_step - 1e-12
                progress && println("  reducing $(continuation_label) step to $(round(step; digits = 5))")
            end
        end

        if !step_success
            if best_trial_alpha > current_alpha + 1e-12
                progress && print_progress_result("$(continuation_label) best unaccepted α=$(round(best_trial_alpha; digits = 4))", best_trial_result)
                return best_trial_alpha, best_trial_result, best_trial_t, best_trial_z
            end
            return current_alpha, final_result, previous_t, previous_z
        end
    end

    return current_alpha, final_result, previous_t, previous_z
end

"""
    better_target_result(lhs, rhs)

Selects the better of two collocation results, prioritizing success and then smaller residual norm.

Input arguments:
- `lhs::CollocationResult`: first candidate.
- `rhs::CollocationResult`: second candidate.

Optional parameters:
- None.

Output:
- Returns one `CollocationResult`.
- No trajectory sizes are changed; the returned object is one of the two inputs.
"""
function better_target_result(lhs::CollocationResult, rhs::CollocationResult)
    if lhs.success != rhs.success
        return lhs.success ? lhs : rhs
    end
    if isfinite(lhs.residual_norm) != isfinite(rhs.residual_norm)
        return isfinite(lhs.residual_norm) ? lhs : rhs
    end
    return lhs.residual_norm <= rhs.residual_norm ? lhs : rhs
end

lambda_continuation_weight(alpha::Real) = Float64(alpha)^3
state_continuation_weight(alpha::Real) = Float64(alpha)^2
state_seed_acceptance_tolerance() = 5e-3

function build_state_tvc_seed(p::ModelParams, steady::SteadyStateResult;
    N::Int = p.N,
    progress::Bool = true)
    q_anchor = continuation_q_anchor(p)
    steady_params = with_initial_conditions(p, steady.k, q_anchor; Λ20 = steady.Λ2)
    steady_result, previous_t, previous_z = solve_collocation_problem(steady_params, steady;
        N = N,
        progress = progress,
        use_mesh_continuation = true,
        terminal_mode = :state_steady_state)
    if !isfinite(steady_result.residual_norm)
        return steady_result, previous_t, previous_z
    end

    best_result = steady_result
    best_t = previous_t
    best_z = previous_z

    progress && println("OptimalWealthTax state seed continuation on Λ20")
    Λ2_alpha, Λ2_result, previous_t, previous_z = continue_initial_conditions(p, steady, previous_t, previous_z;
        N = N,
        progress = progress,
        base_step = 0.05,
        min_step = 1e-4,
        terminal_mode = :state_steady_state,
        label = "state-seed-Λ20",
        endpoint = α -> (steady.k, q_anchor, steady.Λ2 + lambda_continuation_weight(α) * (p.Λ20 - steady.Λ2)))
    if better_target_result(Λ2_result, best_result) === Λ2_result
        best_result = Λ2_result
        best_t = previous_t
        best_z = previous_z
    end
    if Λ2_alpha < 1.0 - 1e-12
        return best_result, best_t, best_z
    end

    after_lambda_t = previous_t
    after_lambda_z = previous_z

    progress && println("OptimalWealthTax state seed joint continuation on k0 and q0")
    kq_alpha, kq_result, kq_t, kq_z = continue_initial_conditions(p, steady, after_lambda_t, after_lambda_z;
        N = N,
        progress = progress,
        acceptance_tolerance = state_seed_acceptance_tolerance(),
        terminal_mode = :state_steady_state,
        label = "state-seed-k0-q0",
        endpoint = α -> (
            steady.k + state_continuation_weight(α) * (p.k0 - steady.k),
            q_anchor + state_continuation_weight(α) * (p.q0 - q_anchor),
            p.Λ20,
        ))
    if better_target_result(kq_result, best_result) === kq_result
        best_result = kq_result
        best_t = kq_t
        best_z = kq_z
    end
    if kq_alpha >= 1.0 - 1e-12 && kq_result.success
        return kq_result, kq_t, kq_z
    end

    progress && println("OptimalWealthTax state seed retry with k0 then q0")
    k_alpha, k_result, k_t, k_z = continue_initial_conditions(p, steady, after_lambda_t, after_lambda_z;
        N = N,
        progress = progress,
        base_step = 0.05,
        min_step = 1e-4,
        terminal_mode = :state_steady_state,
        label = "state-seed-k0",
        endpoint = α -> (steady.k + state_continuation_weight(α) * (p.k0 - steady.k), q_anchor, p.Λ20))
    if better_target_result(k_result, best_result) === k_result
        best_result = k_result
        best_t = k_t
        best_z = k_z
    end
    if k_alpha >= 1.0 - 1e-12
        q_alpha, q_result, q_t, q_z = continue_initial_conditions(p, steady, k_t, k_z;
            N = N,
            progress = progress,
            base_step = 0.05,
            min_step = 1e-4,
            acceptance_tolerance = state_seed_acceptance_tolerance(),
            terminal_mode = :state_steady_state,
            label = "state-seed-q0",
            endpoint = α -> (p.k0, q_anchor + state_continuation_weight(α) * (p.q0 - q_anchor), p.Λ20))
        if better_target_result(q_result, best_result) === q_result
            best_result = q_result
            best_t = q_t
            best_z = q_z
        end
        if q_alpha >= 1.0 - 1e-12 && q_result.success
            return q_result, q_t, q_z
        end
    end

    progress && println("OptimalWealthTax state seed retry with q0 then k0")
    q_first_alpha, q_first_result, q_first_t, q_first_z = continue_initial_conditions(p, steady, after_lambda_t, after_lambda_z;
        N = N,
        progress = progress,
        base_step = 0.05,
        min_step = 1e-4,
        acceptance_tolerance = state_seed_acceptance_tolerance(),
        terminal_mode = :state_steady_state,
        label = "state-seed-q0-first",
        endpoint = α -> (steady.k, q_anchor + state_continuation_weight(α) * (p.q0 - q_anchor), p.Λ20))
    if better_target_result(q_first_result, best_result) === q_first_result
        best_result = q_first_result
        best_t = q_first_t
        best_z = q_first_z
    end
    if q_first_alpha >= 1.0 - 1e-12
        q_first_k_alpha, q_first_k_result, q_first_k_t, q_first_k_z = continue_initial_conditions(p, steady, q_first_t, q_first_z;
            N = N,
            progress = progress,
            base_step = 0.05,
            min_step = 1e-4,
            terminal_mode = :state_steady_state,
            label = "state-seed-k0-second",
            endpoint = α -> (steady.k + state_continuation_weight(α) * (p.k0 - steady.k), p.q0, p.Λ20))
        if better_target_result(q_first_k_result, best_result) === q_first_k_result
            best_result = q_first_k_result
            best_t = q_first_k_t
            best_z = q_first_k_z
        end
        if q_first_k_alpha >= 1.0 - 1e-12 && q_first_k_result.success
            return q_first_k_result, q_first_k_t, q_first_k_z
        end
    end

    return best_result, best_t, best_z
end

"""
    solve_collocation(p=ModelParams(); N=p.N, progress=true, use_continuation=true, use_bvp_refinement=false, use_horizon_continuation=false, terminal_mode=:tvc, use_nested_seed=true)

High-level entry point for solving the `OptimalWealthTax` collocation problem with direct solves, continuation, optional BVP refinement, and optional TVC homotopy.

Input arguments:
- `p::ModelParams = ModelParams()`: model parameters.

Optional parameters:
- `N::Int = p.N`: collocation node count.
- `progress::Bool = true`: print detailed progress information.
- `use_continuation::Bool = true`: enable continuation strategies in initial conditions.
- `use_bvp_refinement::Bool = false`: enable BVP-based refinement attempts.
- `use_horizon_continuation::Bool = false`: enable continuation in the time horizon.
- `terminal_mode::Symbol = :tvc`: terminal closure rule. Supported values include `:costate_steady_state`, `:state_steady_state`, and `:tvc`.
- `use_nested_seed::Bool = true`: allow recursive construction of better seeds through easier closure rules.

Output:
- Returns a `CollocationResult`.
- Every trajectory field in the result has length equal to the final grid size used by the successful or best failed attempt, typically `N`.
"""
function solve_collocation(p::ModelParams = ModelParams(); N::Int = p.N, progress::Bool = true, use_continuation::Bool = true, use_bvp_refinement::Bool = false, use_horizon_continuation::Bool = false, terminal_mode::Symbol = :tvc, use_nested_seed::Bool = true)
    terminal_mode = canonical_terminal_mode(terminal_mode)
    q_anchor = continuation_q_anchor(p)
    steady = find_steady_state(p)
    progress && println("OptimalWealthTax solve start: terminal_mode=$(terminal_mode) T=$(p.T) N=$(N) k0=$(p.k0) q0=$(p.q0) Λ20=$(p.Λ20)")
    progress && println("OptimalWealthTax steady state: k*=$(steady.k) c*=$(steady.c) q*=$(steady.q) Λ1*=$(steady.Λ1) Λ2*=$(steady.Λ2) Λ3*=$(steady.Λ3)")

    if is_tvc_like_terminal_mode(terminal_mode)
        seed_t = nothing
        seed_z = nothing
        homotopy_seed_t = nothing
        homotopy_seed_z = nothing
        failed_target_attempts = CollocationResult[]

        if use_nested_seed
            progress && println("OptimalWealthTax building explicit state-closure seed for TVC solve")
            seed_result, seed_t, seed_z = build_state_tvc_seed(p, steady;
                N = N,
                progress = progress)
            if !isfinite(seed_result.residual_norm)
                seed_t = nothing
                seed_z = nothing
            end
        end

        if seed_z !== nothing
            progress && println("OptimalWealthTax attempting terminal homotopy from state closure to TVC")
            terminal_alpha, terminal_result, terminal_t, terminal_z = continue_terminal_conditions(p, steady, seed_t, seed_z;
                N = N,
                progress = progress,
                base_step = 0.1,
                min_step = 1e-4,
                acceptance_tolerance = tvc_terminal_acceptance_tolerance(p),
                terminal_mode = terminal_mode)
            full_tvc_seed_result = evaluate_candidate(terminal_z, p, steady, terminal_t;
                success = false,
                terminal_mode = terminal_mode)
            push!(failed_target_attempts, full_tvc_seed_result)

            if terminal_alpha >= 1.0 - 1e-12
                progress && println("OptimalWealthTax refining full TVC solve from homotopy seed")
                refined_result, refined_t, refined_z = solve_collocation_problem(p, steady;
                    N = N,
                    progress = progress,
                    initial_t = terminal_t,
                    initial_z = terminal_z,
                    use_mesh_continuation = false,
                    terminal_mode = terminal_mode)
                push!(failed_target_attempts, refined_result)
                if refined_result.success
                    progress && print_progress_result("refined TVC solve", refined_result)
                    return refined_result
                end
                homotopy_seed_t = refined_t
                homotopy_seed_z = refined_z
            else
                homotopy_seed_t = terminal_t
                homotopy_seed_z = terminal_z
            end
        end

        progress && println(seed_z === nothing ?
            "OptimalWealthTax attempting direct TVC solve from collocation guess" :
            "OptimalWealthTax attempting direct TVC solve from state-based seed")
        direct_result, direct_t, direct_z = solve_collocation_problem(p, steady;
            N = N,
            progress = progress,
            initial_t = seed_t,
            initial_z = seed_z,
            use_mesh_continuation = seed_z === nothing,
            terminal_mode = terminal_mode)
        if direct_result.success
            progress && print_progress_result("direct TVC solve", direct_result)
            return direct_result
        end
        progress && print_progress_result("direct TVC solve failed", direct_result)
        push!(failed_target_attempts, direct_result)

        if homotopy_seed_z !== nothing
            progress && println("OptimalWealthTax attempting direct TVC solve from homotopy seed")
            homotopy_result, homotopy_t, homotopy_z = solve_collocation_problem(p, steady;
                N = N,
                progress = progress,
                initial_t = homotopy_seed_t,
                initial_z = homotopy_seed_z,
                use_mesh_continuation = false,
                terminal_mode = terminal_mode)
            if homotopy_result.success
                progress && print_progress_result("homotopy-seeded TVC solve", homotopy_result)
                return homotopy_result
            end
            progress && print_progress_result("homotopy-seeded TVC solve failed", homotopy_result)
            push!(failed_target_attempts, homotopy_result)
            if better_target_result(homotopy_result, direct_result) === homotopy_result
                direct_result = homotopy_result
                direct_t = homotopy_t
                direct_z = homotopy_z
            end
        end

        progress && println("OptimalWealthTax attempting multiple-shooting seed for TVC")
        multiple_shooting_result = solve_multiple_shooting(p;
            progress = progress,
            terminal_mode = terminal_mode,
            use_horizon_continuation = false,
            output_N = max(N, 101),
            segment_length = 2.0)
        push!(failed_target_attempts, multiple_shooting_result)
        if multiple_shooting_result.success
            progress && print_progress_result("multiple-shooting TVC solve", multiple_shooting_result)
            return multiple_shooting_result
        end
        progress && print_progress_result("multiple-shooting TVC solve failed", multiple_shooting_result)
        if better_target_result(multiple_shooting_result, direct_result) === multiple_shooting_result
            direct_result = multiple_shooting_result
            direct_t = multiple_shooting_result.t
            direct_z = pack_solution(multiple_shooting_result)
            progress && println("OptimalWealthTax attempting collocation refinement from multiple-shooting seed")
            ms_refined_result, ms_refined_t, ms_refined_z = solve_collocation_problem(p, steady;
                N = N,
                progress = progress,
                initial_t = multiple_shooting_result.t,
                initial_z = pack_solution(multiple_shooting_result),
                use_mesh_continuation = false,
                terminal_mode = terminal_mode)
            push!(failed_target_attempts, ms_refined_result)
            if ms_refined_result.success
                progress && print_progress_result("multiple-shooting-seeded TVC solve", ms_refined_result)
                return ms_refined_result
            end
            progress && print_progress_result("multiple-shooting-seeded TVC solve failed", ms_refined_result)
            if better_target_result(ms_refined_result, direct_result) === ms_refined_result
                direct_result = ms_refined_result
                direct_t = ms_refined_t
                direct_z = ms_refined_z
            end
        end

        if use_horizon_continuation
            T_stages = default_horizon_stages(p.T)
            progress && println("OptimalWealthTax attempting staged horizon continuation for TVC")
            horizon_result, horizon_t, horizon_z = continue_horizon_stages(p, steady, T_stages;
                N = N,
                progress = progress,
                terminal_mode = terminal_mode,
                acceptance_tolerance = 5e-2)
            if horizon_reached_target(horizon_result, p.T) && horizon_result.success
                progress && print_progress_result("staged horizon continuation", horizon_result)
                return horizon_result
            end
            if horizon_reached_target(horizon_result, p.T)
                progress && print_progress_result("staged horizon continuation failed", horizon_result)
                push!(failed_target_attempts, horizon_result)
                if better_target_result(horizon_result, direct_result) === horizon_result
                    direct_result = horizon_result
                    direct_t = horizon_t
                    direct_z = horizon_z
                end
            else
                progress && println("OptimalWealthTax staged horizon continuation stopped early at T=$(round(horizon_result.t[end]; digits = 4))")
            end
        end

        if use_bvp_refinement
            push!(failed_target_attempts, solve_bvp_problem(p, steady; N = N, progress = progress, initial_t = direct_t, initial_z = direct_z, terminal_mode = terminal_mode))
        end

        best_result = reduce(better_target_result, failed_target_attempts)
        progress && print_progress_result("best TVC result after staged seed", best_result)
        return best_result
    end

    seed_t = nothing
    seed_z = nothing
    if terminal_mode == :state_steady_state && use_nested_seed
        progress && println("OptimalWealthTax building seed from costate_steady_state closure")
        seed_result = solve_collocation(p;
            N = N,
            progress = progress,
            use_continuation = use_continuation,
            use_bvp_refinement = false,
            use_horizon_continuation = use_horizon_continuation,
            terminal_mode = :costate_steady_state,
            use_nested_seed = true)
        if isfinite(seed_result.residual_norm)
            seed_t = seed_result.t
            seed_z = pack_solution(seed_result)
            progress && print_progress_result("seed result", seed_result)
        end
    end

    steady_params = with_initial_conditions(p, steady.k, q_anchor; Λ20 = steady.Λ2)
    progress && println("OptimalWealthTax solving steady-state anchor problem")
    steady_result, previous_t, previous_z = solve_collocation_problem(steady_params, steady; N = N, progress = progress, use_mesh_continuation = true, terminal_mode = terminal_mode)
    if !steady_result.success
        progress && print_progress_result("steady-state anchor failed", steady_result)
        return steady_result
    end

    target_gap = max(
        abs(p.k0 - steady.k) / max(abs(steady.k), 1.0),
        abs(p.q0 - q_anchor) / max(abs(q_anchor), 1.0),
        abs(p.Λ20 - steady.Λ2) / max(abs(steady.Λ2), 1.0),
    )
    if target_gap <= 1e-12
        return steady_result
    end

    failed_target_attempts = CollocationResult[]

    progress && println("OptimalWealthTax attempting direct target solve")
    direct_result, direct_t, direct_z = solve_collocation_problem(p, steady;
        N = N,
        progress = progress,
        initial_t = seed_t,
        initial_z = seed_z,
        use_mesh_continuation = true,
        terminal_mode = terminal_mode)
    if direct_result.success
        progress && print_progress_result("direct target solve", direct_result)
        return direct_result
    end
    progress && print_progress_result("direct target solve failed", direct_result)
    push!(failed_target_attempts, direct_result)

    progress && println("OptimalWealthTax attempting ordered initial-condition continuation")
    initial_t = previous_t
    initial_z = previous_z

    progress && println("OptimalWealthTax ordered initial continuation on Λ20")
    initial_Λ2_alpha, initial_result, initial_t, initial_z = continue_initial_conditions(p, steady, initial_t, initial_z;
        N = N,
        progress = progress,
        base_step = 0.05,
        min_step = 1e-4,
        terminal_mode = terminal_mode,
        label = "initial-Λ20",
        endpoint = α -> (steady.k, q_anchor, steady.Λ2 + lambda_continuation_weight(α) * (p.Λ20 - steady.Λ2)))
    if initial_Λ2_alpha >= 1.0 - 1e-12
        progress && println("OptimalWealthTax ordered initial continuation on k0")
        initial_k_alpha, initial_result, initial_t, initial_z = continue_initial_conditions(p, steady, initial_t, initial_z;
            N = N,
            progress = progress,
            base_step = 0.05,
            min_step = 1e-4,
            terminal_mode = terminal_mode,
            label = "initial-k0",
            endpoint = α -> (steady.k + α * (p.k0 - steady.k), q_anchor, p.Λ20))
        if initial_k_alpha >= 1.0 - 1e-12
            progress && println("OptimalWealthTax ordered initial continuation on q0")
            initial_q_alpha, initial_result, initial_t, initial_z = continue_initial_conditions(p, steady, initial_t, initial_z;
                N = N,
                progress = progress,
                base_step = 0.05,
                min_step = 1e-4,
                acceptance_tolerance = 5e-2,
                terminal_mode = terminal_mode,
                label = "initial-q0",
                endpoint = α -> (p.k0, q_anchor + α * (p.q0 - q_anchor), p.Λ20))
            if initial_q_alpha >= 1.0 - 1e-12 && initial_result.success
                progress && print_progress_result("ordered initial continuation reached target", initial_result)
                return better_target_result(direct_result, initial_result)
            end
        end
    end
    initial_result = evaluate_candidate(initial_z, p, steady, initial_t; success = false, terminal_mode = terminal_mode)
    progress && print_progress_result("ordered initial continuation partial candidate", initial_result)
    push!(failed_target_attempts, initial_result)
    if better_target_result(initial_result, direct_result) === initial_result
        direct_t, direct_z = initial_t, initial_z
    end

    if use_horizon_continuation
        progress && println("OptimalWealthTax attempting horizon continuation")
        horizon_result, horizon_t, horizon_z = continue_horizon(p, steady; N = N, progress = progress, target_T = p.T, terminal_mode = terminal_mode)
        if horizon_reached_target(horizon_result, p.T) && horizon_result.success
            progress && print_progress_result("horizon continuation", horizon_result)
            return horizon_result
        end
        if horizon_reached_target(horizon_result, p.T)
            progress && print_progress_result("horizon continuation failed", horizon_result)
            push!(failed_target_attempts, horizon_result)
            if better_target_result(horizon_result, direct_result) === horizon_result
                direct_t, direct_z = horizon_t, horizon_z
            end
        else
            progress && println("OptimalWealthTax horizon continuation stopped early at T=$(round(horizon_result.t[end]; digits = 4))")
        end
    end

    if use_bvp_refinement
        push!(failed_target_attempts, solve_bvp_problem(p, steady; N = N, progress = progress, initial_t = direct_t, initial_z = direct_z, terminal_mode = terminal_mode))
    end

    if !use_continuation
        best_result = reduce(better_target_result, failed_target_attempts)
        progress && print_progress_result("best result without continuation", best_result)
        return best_result
    end

    progress && println("OptimalWealthTax attempting diag continuation")
    diag_alpha, diag_result, previous_t, previous_z = continue_initial_conditions(p, steady, previous_t, previous_z;
        N = N,
        progress = progress,
        terminal_mode = terminal_mode,
        label = "diag",
        endpoint = α -> (
            steady.k + state_continuation_weight(α) * (p.k0 - steady.k),
            q_anchor + state_continuation_weight(α) * (p.q0 - q_anchor),
            steady.Λ2 + α * (p.Λ20 - steady.Λ2),
        ))
    if diag_alpha >= 1.0 - 1e-12
        progress && print_progress_result("diag continuation reached target", diag_result)
        return diag_result
    end

    if use_bvp_refinement
        push!(failed_target_attempts, solve_bvp_problem(p, steady; N = N, progress = progress, initial_t = previous_t, initial_z = previous_z, terminal_mode = terminal_mode))
    end

    progress && println("OptimalWealthTax restarting from steady-state anchor for ordered continuation")
    steady_result, previous_t, previous_z = solve_collocation_problem(steady_params, steady; N = N, progress = progress, use_mesh_continuation = true, terminal_mode = terminal_mode)
    if !steady_result.success
        progress && print_progress_result("steady-state anchor restart failed", steady_result)
        return steady_result
    end

    progress && println("OptimalWealthTax ordered continuation on Λ20")
    Λ2_alpha, Λ2_result, previous_t, previous_z = continue_initial_conditions(p, steady, previous_t, previous_z;
        N = N,
        progress = progress,
        terminal_mode = terminal_mode,
        label = "Λ20",
        endpoint = α -> (steady.k, q_anchor, steady.Λ2 + lambda_continuation_weight(α) * (p.Λ20 - steady.Λ2)))
    if Λ2_alpha < 1.0 - 1e-12
        push!(failed_target_attempts, evaluate_candidate(previous_z, p, steady, previous_t; success = false, terminal_mode = terminal_mode))
        if use_bvp_refinement
            push!(failed_target_attempts, solve_bvp_problem(p, steady; N = N, progress = progress, initial_t = previous_t, initial_z = previous_z, terminal_mode = terminal_mode))
        end
        best_result = reduce(better_target_result, failed_target_attempts)
        progress && print_progress_result("ordered continuation stopped on Λ20", best_result)
        return best_result
    end

    after_lambda_t = previous_t
    after_lambda_z = previous_z

    progress && println("OptimalWealthTax ordered joint continuation on k0 and q0")
    kq_alpha, kq_result, kq_t, kq_z = continue_initial_conditions(p, steady, after_lambda_t, after_lambda_z;
        N = N,
        progress = progress,
        acceptance_tolerance = 5e-2,
        terminal_mode = terminal_mode,
        label = "k0-q0",
        endpoint = α -> (
            steady.k + α * (p.k0 - steady.k),
            q_anchor + α * (p.q0 - q_anchor),
            p.Λ20,
        ))
    kq_branch_result = kq_alpha >= 1.0 - 1e-12 ? kq_result : evaluate_candidate(kq_z, p, steady, kq_t; success = false, terminal_mode = terminal_mode)
    if kq_alpha >= 1.0 - 1e-12 && kq_result.success
        progress && print_progress_result("ordered joint continuation reached target", kq_result)
        return kq_result
    end
    progress && println("OptimalWealthTax retrying full target from joint k0-q0 branch")
    kq_target_result, kq_target_t, kq_target_z = solve_collocation_problem(p, steady;
        N = N,
        progress = progress,
        initial_t = kq_t,
        initial_z = kq_z,
        use_mesh_continuation = false,
        terminal_mode = terminal_mode)
    if kq_target_result.success
        progress && print_progress_result("joint-branch target solve", kq_target_result)
        return kq_target_result
    end
    progress && print_progress_result("joint-branch target solve failed", kq_target_result)

    hybrid_kq_target_result = kq_target_result
    hybrid_kq_target_t = kq_target_t
    hybrid_kq_target_z = kq_target_z
    if terminal_mode == :costate_steady_state
        progress && println("OptimalWealthTax trying state-closure k0-q0 continuation as seed for costate target")
        state_kq_alpha, state_kq_result, state_kq_t, state_kq_z = continue_initial_conditions(p, steady, after_lambda_t, after_lambda_z;
            N = N,
            progress = progress,
            acceptance_tolerance = 5e-2,
            terminal_mode = :state_steady_state,
            label = "state-k0-q0",
            endpoint = α -> (
                steady.k + α * (p.k0 - steady.k),
                q_anchor + α * (p.q0 - q_anchor),
                p.Λ20,
            ))
        state_kq_branch_result = state_kq_alpha >= 1.0 - 1e-12 ? state_kq_result : evaluate_candidate(state_kq_z, p, steady, state_kq_t;
            success = false,
            terminal_mode = :state_steady_state)
        progress && print_progress_result("state-closure k0-q0 branch", state_kq_branch_result)

        progress && println("OptimalWealthTax retrying costate target from state-closure k0-q0 branch")
        hybrid_kq_target_result, hybrid_kq_target_t, hybrid_kq_target_z = solve_collocation_problem(p, steady;
            N = N,
            progress = progress,
            initial_t = state_kq_t,
            initial_z = state_kq_z,
            use_mesh_continuation = false,
            terminal_mode = :costate_steady_state)
        if hybrid_kq_target_result.success
            progress && print_progress_result("state-seeded costate target solve", hybrid_kq_target_result)
            return hybrid_kq_target_result
        end
        progress && print_progress_result("state-seeded costate target solve failed", hybrid_kq_target_result)
    end

    progress && println("OptimalWealthTax ordered continuation on k0")
    k_alpha, k_result, k_t, k_z = continue_initial_conditions(p, steady, previous_t, previous_z;
        N = N,
        progress = progress,
        terminal_mode = terminal_mode,
        label = "k0",
        endpoint = α -> (steady.k + state_continuation_weight(α) * (p.k0 - steady.k), q_anchor, p.Λ20))
    k_branch_result = k_alpha >= 1.0 - 1e-12 ? k_result : evaluate_candidate(k_z, p, steady, k_t; success = false, terminal_mode = terminal_mode)

    q_alpha = 0.0
    q_result = k_branch_result
    q_t = k_t
    q_z = k_z
    if k_alpha >= 1.0 - 1e-12
        progress && println("OptimalWealthTax ordered continuation on q0")
        q_alpha, q_result, q_t, q_z = continue_initial_conditions(p, steady, k_t, k_z;
            N = N,
            progress = progress,
            acceptance_tolerance = 5e-2,
            terminal_mode = terminal_mode,
            label = "q0",
            endpoint = α -> (p.k0, q_anchor + state_continuation_weight(α) * (p.q0 - q_anchor), p.Λ20))
        if q_alpha >= 1.0 - 1e-12 && q_result.success
            progress && print_progress_result("ordered continuation reached target", q_result)
            return q_result
        end
    end

    best_branch_result = reduce(better_target_result, (hybrid_kq_target_result, kq_target_result, kq_branch_result, k_branch_result, q_result))
    best_branch_t = best_branch_result === hybrid_kq_target_result ? hybrid_kq_target_t : (best_branch_result === kq_target_result ? kq_target_t : (best_branch_result === kq_branch_result ? kq_t : (best_branch_result === q_result ? q_t : k_t)))
    best_branch_z = best_branch_result === hybrid_kq_target_result ? hybrid_kq_target_z : (best_branch_result === kq_target_result ? kq_target_z : (best_branch_result === kq_branch_result ? kq_z : (best_branch_result === q_result ? q_z : k_z)))

    progress && println("OptimalWealthTax retrying ordered continuation with q0 before k0")
    q_first_alpha, q_first_result, q_first_t, q_first_z = continue_initial_conditions(p, steady, after_lambda_t, after_lambda_z;
        N = N,
        progress = progress,
        acceptance_tolerance = 5e-2,
        terminal_mode = terminal_mode,
        label = "q0-first",
        endpoint = α -> (steady.k, q_anchor + state_continuation_weight(α) * (p.q0 - q_anchor), p.Λ20))
    q_first_branch_result = q_first_alpha >= 1.0 - 1e-12 ? q_first_result : evaluate_candidate(q_first_z, p, steady, q_first_t; success = false, terminal_mode = terminal_mode)
    if better_target_result(q_first_branch_result, best_branch_result) === q_first_branch_result
        best_branch_result = q_first_branch_result
        best_branch_t = q_first_t
        best_branch_z = q_first_z
    end

    if q_first_alpha >= 1.0 - 1e-12
        progress && println("OptimalWealthTax q-first continuation on k0")
        q_first_k_alpha, q_first_k_result, q_first_k_t, q_first_k_z = continue_initial_conditions(p, steady, q_first_t, q_first_z;
            N = N,
            progress = progress,
            terminal_mode = terminal_mode,
            label = "k0-second",
            endpoint = α -> (steady.k + state_continuation_weight(α) * (p.k0 - steady.k), p.q0, p.Λ20))
        q_first_k_branch_result = q_first_k_alpha >= 1.0 - 1e-12 ? q_first_k_result : evaluate_candidate(q_first_k_z, p, steady, q_first_k_t; success = false, terminal_mode = terminal_mode)
        if q_first_k_alpha >= 1.0 - 1e-12 && q_first_k_result.success
            progress && print_progress_result("q-first ordered continuation reached target", q_first_k_result)
            return q_first_k_result
        end
        if better_target_result(q_first_k_branch_result, best_branch_result) === q_first_k_branch_result
            best_branch_result = q_first_k_branch_result
            best_branch_t = q_first_k_t
            best_branch_z = q_first_k_z
        end
    end

    push!(failed_target_attempts, best_branch_result)
    previous_t = best_branch_t
    previous_z = best_branch_z

    progress && println("OptimalWealthTax retrying full target from ordered-continuation branch")
    target_from_branch, branch_t, branch_z = solve_collocation_problem(p, steady;
        N = N,
        progress = progress,
        initial_t = previous_t,
        initial_z = previous_z,
        use_mesh_continuation = false,
        terminal_mode = terminal_mode)
    if target_from_branch.success
        progress && print_progress_result("branch-seeded target solve", target_from_branch)
        return target_from_branch
    end
    progress && print_progress_result("branch-seeded target solve failed", target_from_branch)
    push!(failed_target_attempts, target_from_branch)
    direct_t, direct_z = branch_t, branch_z

    if terminal_mode == :state_steady_state
        fallback_seed_t = seed_t
        fallback_seed_z = seed_z
        if fallback_seed_z === nothing
            progress && println("OptimalWealthTax building fallback seed from costate_steady_state closure")
            costate_seed_result = solve_collocation(p;
                N = N,
                progress = progress,
                use_continuation = use_continuation,
                use_bvp_refinement = false,
                use_horizon_continuation = use_horizon_continuation,
                terminal_mode = :costate_steady_state,
                use_nested_seed = false)
            if isfinite(costate_seed_result.residual_norm)
                fallback_seed_t = costate_seed_result.t
                fallback_seed_z = pack_solution(costate_seed_result)
            end
        end
        if fallback_seed_z !== nothing
            progress && println("OptimalWealthTax attempting terminal homotopy from costate to state closure")
            state_alpha, state_homotopy_result, state_t, state_z = continue_terminal_conditions(p, steady, fallback_seed_t, fallback_seed_z;
                N = N,
                progress = progress,
                base_step = 0.1,
                min_step = 1e-4,
                acceptance_tolerance = 5e-2,
                terminal_mode = :state_steady_state)
            push!(failed_target_attempts, evaluate_candidate(state_z, p, steady, state_t;
                success = false,
                terminal_mode = :state_steady_state,
                terminal_alpha = state_alpha))

            if state_alpha >= 1.0 - 1e-12
                progress && println("OptimalWealthTax refining full state solve from costate homotopy seed")
                refined_state_result, refined_state_t, refined_state_z = solve_collocation_problem(p, steady;
                    N = N,
                    progress = progress,
                    initial_t = state_t,
                    initial_z = state_z,
                    use_mesh_continuation = false,
                    terminal_mode = :state_steady_state)
                if refined_state_result.success
                    progress && print_progress_result("costate-to-state homotopy", refined_state_result)
                    return refined_state_result
                end
                progress && print_progress_result("costate-to-state homotopy failed", refined_state_result)
                push!(failed_target_attempts, refined_state_result)
                direct_t, direct_z = refined_state_t, refined_state_z
            elseif better_target_result(state_homotopy_result, direct_result) === state_homotopy_result
                direct_t, direct_z = state_t, state_z
            end
        end
    end

    if use_bvp_refinement
        push!(failed_target_attempts, solve_bvp_problem(p, steady; N = N, progress = progress, initial_t = previous_t, initial_z = previous_z, terminal_mode = terminal_mode))
    end
    best_result = reduce(better_target_result, failed_target_attempts)
    progress && print_progress_result("best result after all attempts", best_result)
    return best_result
end