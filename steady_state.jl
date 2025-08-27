# Steady state computation for the Optimal Redistributive Capital Taxation Model

# This module computes the steady state (k*, c*, λ*, μ*) under the positive-interest
# regime r̃* = ρ, and checks complementarity via r̃ = max{0, A(1-η)k^{θ-1} - δ - γ/(λ k)}.
#
# We use the analytical expression provided for k*:
#   k* = (A θ / (ρ + δ))^(1/(1-θ))
# and then back out the remaining steady-state variables assuming r̃* = ρ:
#   c* = ρ k* + A η (k*)^θ
#   x* = A(1-η)(k*)^θ - (δ+ρ)k*
#   λ* = γ / (k* [A(1-η)(k*)^{θ-1} - δ - ρ])
#   μ* = ( (c*)^{-β} - λ* ) / ρ

module SteadyState

export find_steady_state, SteadyStateResult


struct SteadyStateResult
    k::Float64
    c::Float64
    λ::Float64
    μ::Float64
    r_tilde::Float64
    x::Float64
    tau_k::Float64
end

"""
    find_steady_state(p)

Compute steady state using the analytical expression for k*:
    k* = (A θ / (ρ + δ))^(1/(1-θ))
Assumes positive-interest steady state r̃* = ρ > 0. Returns `SteadyStateResult`
and validates interior conditions (A(1-η)k^{θ-1} - δ - ρ > 0 and x* > 0).
"""
function find_steady_state(p)
    ρ, A, θ, η, β, δ, γ = p.ρ, p.A, p.θ, p.η, p.β, p.δ, p.γ

    # New formula: k* = (A θ / (ρ + δ))^(1/(1-θ))
    base = A * θ / (ρ + δ)
    if base <= 0
        error("Invalid parameters for k*: A*θ/(ρ+δ) must be > 0, got $(base)")
    end
    kstar = base^(1 / (1 - θ))

    # Back out remaining steady-state values under r̃* = ρ
    rstar = ρ
    c = rstar * kstar + A * η * kstar^θ
    denom_k = A * (1 - η) * kstar^(θ - 1) - δ - rstar
    if denom_k <= 0
        error("Interior steady state invalid: A(1-η)k^{θ-1} - δ - ρ = $(denom_k) ≤ 0")
    end
    x = A * (1 - η) * kstar^θ - (δ + rstar) * kstar
    if x <= 0
        error("Net capital accumulation x* ≤ 0 at k*: x*=$(x)")
    end
    λ = γ / (kstar * denom_k)
    μ = (c^(-β) - λ) / rstar
    r_tilde = rstar

    fnames = fieldnames(typeof(p))
    tau_k = ((:r in fnames) && p.r > 0) ? max(0.0, 1.0 - r_tilde / p.r) : 0.0

    return SteadyStateResult(kstar, c, λ, μ, r_tilde, x, tau_k)
end

end # module
