function production_scale(p::ModelParams)
    return p.A * p.n^p.η * p.l^(1.0 - p.θ - p.η)
end

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

smooth_positive_part(z::Real, ε::Real) = 0.5 * (z + sqrt(z * z + ε * ε))

function foc_implied_controls(y::AbstractVector{<:Real}, p::ModelParams)
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
    if !isfinite(denom)
        return nothing
    end

    # KKT-implied effective return: take the interior maximizer when feasible,
    # then enforce the admissible set r_tilde >= 0.
    r_unconstrained = if denom > p.min_positive
        resource_term / sum_eff - p.γ / denom
    else
        -Inf
    end
    r_tilde = smooth_positive_part(r_unconstrained, p.min_positive)

    if !isfinite(r_tilde)
        return nothing
    end

    x = resource_term - sum_eff * r_tilde
    x = (!isfinite(x) || x <= p.min_positive) ? p.min_positive : x
    if !isfinite(x)
        return nothing
    end

    return (; r_tilde, x, terms...)
end

function dynamics(y::AbstractVector{<:Real}, p::ModelParams)
    controls = foc_implied_controls(y, p)
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