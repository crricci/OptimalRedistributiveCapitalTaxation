# Steady state computation for the Optimal Redistributive Capital Taxation model.
#
# The updated system uses 4D dynamics in (k, c, lambda, mu) together with the
# algebraic FOC lambda + mu * c / (beta * k) = gamma / x and
# r_tilde = A * (1 - eta) * k^(theta - 1) - delta - gamma / (lambda * k).
# At steady state we impose r_tilde = rho.
#
# This implementation keeps the analytic formula for k_star and then recovers the
# remaining steady-state objects from the static equilibrium conditions.

module SteadyState
"""
    analytic_steady_state(p)

Return the analytic steady state from the closed-form k_star formula:
    k_star = (((rho + delta) * (theta + gamma) + gamma * rho * theta) /
            (theta * A * (1 + gamma) * (1 - eta)))^(1 / (theta - 1))

Then derive:
    r_tilde_star = rho
    x_star = A * (1 - eta) * k_star^theta - (delta + r_tilde_star) * k_star
    lambda_star + mu_star * c_star / (beta * k_star) = gamma / x_star
    mu_star * rho - c_star^(-beta) + lambda_star = 0
    c_star = rho * k_star + A * eta * k_star^theta
    lambda_star * (rho - r_tilde_star - A * theta * eta * k_star^(theta - 1)) =
            (gamma / x_star) * (A * theta * (1 - eta) * k_star^(theta - 1) - delta - r_tilde_star)

The last equation gives lambda_star, then mu_star follows from the static FOC.
"""
function analytic_steady_state(p)
    ρ, A, θ, η, β, δ, γ = p.ρ, p.A, p.θ, p.η, p.β, p.δ, p.γ
    num = (ρ+δ)*(θ+γ) + γ*ρ*θ
    den = θ * A * (1+γ) * (1-η)
    base = num / den
    if base <= 0
        error("Invalid base for k*: num/den <= 0 (num=$(num), den=$(den))")
    end
    kstar = base^(1/(θ-1))
    r_tilde = ρ
    x = A*(1-η)*kstar^θ - (δ + r_tilde)*kstar
    if x <= 0
        error("x* <= 0 with analytic k*: x=$(x)")
    end
    c = r_tilde*kstar + A*η*kstar^θ
    # Compute lambda_star from the lambda stationarity condition.
    T = A*θ*(1-η)*kstar^(θ-1) - δ - r_tilde
    S = ρ - r_tilde - A*θ*η*kstar^(θ-1)
    # lambda_star * S = (gamma / x) * T.
    if abs(S) < 1e-14
        error("Singular S in lambda formula (S~=0)")
    end
    λ = (γ/x) * T / S
    μ = (c^(-β) - λ)/ρ
    gross_return = A * θ * (1 - η) * kstar^(θ - 1)
    denom = gross_return - δ
    tau_k = abs(denom) > 1e-12 ? (1.0 - r_tilde / denom) : 0.0
    return SteadyStateResult(kstar, c, λ, μ, r_tilde, x, tau_k)
end

export find_steady_state, SteadyStateResult


"""
    SteadyStateResult

Struct storing the steady state of the `NoWealthTaxation` model.

Fields:
- `k`, `c`, `lambda`, `mu`, `r_tilde`, `x`, `tau_k`: all are scalar `Float64` values.

Dimensions:
- Each field has size `1 x 1`; no time trajectories are stored here.
"""
struct SteadyStateResult
    k::Float64
    c::Float64
    λ::Float64
    μ::Float64
    r_tilde::Float64
    x::Float64
    tau_k::Float64
end

const find_steady_state = analytic_steady_state

end # module
