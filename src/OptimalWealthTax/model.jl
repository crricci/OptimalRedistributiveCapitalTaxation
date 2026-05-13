"""
    production_scale(p)

Computes the scale factor of the Cobb-Douglas production function after labor and population are fixed.

Input arguments:
- `p::ModelParams`: model parameters.

Optional parameters:
- None.

Output:
- Returns a scalar `Float64`.
- The returned quantity has size `1 x 1` and is reused in all production derivatives.
"""
function production_scale(p::ModelParams)
    return p.A * p.n^p.η * p.l^(1.0 - p.θ - p.η)
end

"""
    production_terms(k, p)

Evaluates production and its derivatives with respect to capital and effective labor at the point `k`.

Input arguments:
- `k::Real`: current capital, scalar.
- `p::ModelParams`: model parameters.

Optional parameters:
- None.

Output:
- Returns a named tuple with scalar fields `F`, `Fk`, `Fn`, `Fkk`, and `Fnk`.
- Each field has size `1 x 1`.
"""
function production_terms(k::Real, p::ModelParams)
    scale = production_scale(p)
    labor_scale = p.A * p.l^(1.0 - p.θ - p.η)
    F = scale * k^p.θ
    Fk = scale * p.θ * k^(p.θ - 1.0)
    Fn = labor_scale * p.η * k^p.θ * p.n^(p.η - 1.0)
    Fkk = scale * p.θ * (p.θ - 1.0) * k^(p.θ - 2.0)
    Fnk = labor_scale * p.θ * p.η * k^(p.θ - 1.0) * p.n^(p.η - 1.0)
    return (; F, Fk, Fn, Fkk, Fnk)
end

function control_reconstruction_terms(y::AbstractVector{<:Real}, p::ModelParams)
    k, c, q, Λ1, Λ2 = y[1], y[2], y[3], y[4], y[5]
    if !(isfinite(k) && isfinite(c) && isfinite(q) && isfinite(Λ1) && isfinite(Λ2))
        return nothing
    end

    k_eff = max(k, p.min_positive)
    c_eff = max(c, p.min_positive)
    sum_eff = max(k + q, p.min_positive)
    terms = production_terms(k_eff, p)
    resource_term = terms.F - p.δ * k_eff - terms.Fn + q * (terms.Fk - p.δ)
    denom = Λ1 * sum_eff + Λ2 * c_eff / p.β
    if !(isfinite(resource_term) && isfinite(denom))
        return nothing
    end

    r_unconstrained = if denom > p.min_positive
        resource_term / sum_eff - p.γ / denom
    else
        -Inf
    end

    return (; k_eff, c_eff, sum_eff, resource_term, denom, r_unconstrained, terms...)
end

function control_multiplier(base, r_tilde, p::ModelParams)
    x_unclamped = base.resource_term - base.sum_eff * r_tilde
    x_eff = max(x_unclamped, p.min_positive)
    multiplier = p.γ * base.sum_eff / x_eff - base.denom
    return multiplier, x_eff
end

smooth_fischer_burmeister(a, b, ε) = sqrt(a^2 + b^2 + 2.0 * ε^2) - a - b

function clamp_feasible_r_tilde(y::AbstractVector{<:Real}, p::ModelParams, r_tilde::Real)
    base = control_reconstruction_terms(y, p)
    if base === nothing || !isfinite(r_tilde)
        return NaN
    end

    if base.resource_term <= p.min_positive
        return 0.0
    end

    r_upper = max((base.resource_term - p.min_positive) / base.sum_eff, 0.0)
    return clamp(r_tilde, 0.0, r_upper)
end

function controls_from_r_tilde(y::AbstractVector{<:Real}, p::ModelParams, r_tilde::Real)
    base = control_reconstruction_terms(y, p)
    if base === nothing || !isfinite(r_tilde) || r_tilde < 0.0
        return nothing
    end

    x = base.resource_term - base.sum_eff * r_tilde
    if !isfinite(x) || x <= p.min_positive
        return nothing
    end

    return (; r_tilde = Float64(r_tilde), x, F = base.F, Fk = base.Fk, Fn = base.Fn, Fkk = base.Fkk, Fnk = base.Fnk)
end

function control_complementarity_residual(y::AbstractVector{<:Real}, p::ModelParams, r_tilde::Real)
    base = control_reconstruction_terms(y, p)
    if base === nothing || !isfinite(r_tilde) || r_tilde < 0.0
        return NaN
    end

    controls = controls_from_r_tilde(y, p, r_tilde)
    if controls === nothing
        return NaN
    end

    multiplier, _ = control_multiplier(base, r_tilde, p)
    if !isfinite(multiplier)
        return NaN
    end

    return smooth_fischer_burmeister(r_tilde, multiplier, max(p.control_complementarity_smoothing, 0.0))
end

function complementarity_implied_control(base, p::ModelParams)
    if !(isfinite(base.resource_term) && isfinite(base.sum_eff) && isfinite(base.denom))
        return nothing
    end

    if base.resource_term <= p.min_positive
        return 0.0
    end

    r_upper = max((base.resource_term - p.min_positive) / base.sum_eff, 0.0)
    r_tilde = clamp(max(base.r_unconstrained, 0.0), 0.0, r_upper)
    ε = max(p.control_complementarity_smoothing, 0.0)

    for _ in 1:8
        multiplier, x_eff = control_multiplier(base, r_tilde, p)
        radius = sqrt(r_tilde^2 + multiplier^2 + 2.0 * ε^2)
        phi = radius - r_tilde - multiplier
        multiplier_prime = p.γ * base.sum_eff^2 / x_eff^2
        phi_prime = (r_tilde + multiplier * multiplier_prime) / max(radius, p.min_positive) - 1.0 - multiplier_prime
        if !isfinite(phi) || !isfinite(phi_prime)
            return nothing
        end
        step = phi_prime == 0 ? zero(phi) : phi / phi_prime
        candidate = clamp(r_tilde - step, 0.0, r_upper)
        r_tilde = isfinite(candidate) ? candidate : r_tilde
    end

    return r_tilde
end

"""
    foc_implied_controls(y, p)

Reconstructs the implied controls `r_tilde` and `x` from the current state-costate vector and the first-order conditions.

Input arguments:
- `y::AbstractVector{<:Real}`: state-costate vector. The first five entries must be `(k, c, q, Λ1, Λ2)`; if the vector has length 6 the sixth entry is ignored.
- `p::ModelParams`: model parameters.

Optional parameters:
- None.

Output:
- Returns `nothing` when the inputs are numerically invalid.
- Otherwise returns a named tuple with scalar fields `r_tilde`, `x`, `F`, `Fk`, `Fn`, `Fkk`, and `Fnk`.
- All returned components have size `1 x 1`.
"""
function foc_implied_controls(y::AbstractVector{<:Real}, p::ModelParams; active_bound::Union{Nothing, Bool} = nothing)
    base = control_reconstruction_terms(y, p)
    if base === nothing
        return nothing
    end

    # KKT-implied effective return: take the interior maximizer when feasible,
    # then enforce the admissible set r_tilde >= 0.
    r_tilde = if active_bound === true
        0.0
    elseif active_bound === false
        if !isfinite(base.r_unconstrained) || base.r_unconstrained <= 0.0
            return nothing
        end
        base.r_unconstrained
    elseif p.control_kkt_mode == :fischer_burmeister
        candidate = complementarity_implied_control(base, p)
        if candidate === nothing
            return nothing
        end
        candidate
    elseif p.control_kkt_mode == :closed_form
        if isfinite(base.r_unconstrained)
            if p.control_bound_smoothing > 0.0
                # Smooth lower-bound projection for the active-bound regime r_tilde >= 0.
                0.5 * (base.r_unconstrained + sqrt(base.r_unconstrained^2 + p.control_bound_smoothing^2))
            else
                max(0.0, base.r_unconstrained)
            end
        else
            0.0
        end
    else
        error("Unsupported control_kkt_mode=$(p.control_kkt_mode)")
    end

    if !isfinite(r_tilde)
        return nothing
    end

    x = base.resource_term - base.sum_eff * r_tilde
    x = (!isfinite(x) || x <= p.min_positive) ? p.min_positive : x
    if !isfinite(x)
        return nothing
    end

    return (; r_tilde, x, F = base.F, Fk = base.Fk, Fn = base.Fn, Fkk = base.Fkk, Fnk = base.Fnk)
end

"""
    dynamics(y, p)

Evaluates the dynamic system of the optimal-control problem for `(k, c, q, Λ1, Λ2, Λ3)`.

Input arguments:
- `y::AbstractVector{<:Real}`: vector of length 6 ordered as `(k, c, q, Λ1, Λ2, Λ3)`.
- `p::ModelParams`: model parameters.

Optional parameters:
- None.

Output:
- Returns a `Vector{Float64}` of length 6 containing `(dk, dc, dq, dΛ1, dΛ2, dΛ3)`.
- If the controls are not numerically defined, it returns a vector of six `NaN` values.
"""
function dynamics(y::AbstractVector{<:Real}, p::ModelParams; active_bound::Union{Nothing, Bool} = nothing, r_override = nothing)
    controls = r_override === nothing ?
        foc_implied_controls(y, p; active_bound = active_bound) :
        controls_from_r_tilde(y, p, r_override)
    if controls === nothing
        return fill(NaN, 6)
    end

    k, c, q, Λ1, Λ2, Λ3 = y
    k_eff = max(k, p.min_positive)
    c_eff = max(c, p.min_positive)
    r_tilde = controls.r_tilde
    x = controls.x
    Fk = controls.Fk
    Fn = controls.Fn
    Fkk = controls.Fkk
    Fnk = controls.Fnk

    dk = r_tilde * k_eff + q * (r_tilde - (Fk - p.δ)) + Fn - c_eff
    dc = (c_eff / p.β) * (r_tilde - p.ρ)
    dq = q * (Fk - p.δ) - Fn
    dΛ1 = Λ1 * (p.ρ - r_tilde + q * Fkk - Fnk) - Λ3 * (q * Fkk - Fnk) - (p.γ / x) * (Fk - p.δ - r_tilde + q * Fkk - Fnk)
    dΛ2 = Λ2 * (p.ρ - (r_tilde - p.ρ) / p.β) - c_eff^(-p.β) + Λ1
    dΛ3 = Λ3 * (p.ρ - (Fk - p.δ)) - Λ1 * (r_tilde - (Fk - p.δ)) - (p.γ / x) * (Fk - p.δ - r_tilde)

    return [dk, dc, dq, dΛ1, dΛ2, dΛ3]
end