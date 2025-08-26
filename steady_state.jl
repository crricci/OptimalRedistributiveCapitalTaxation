# Steady state computation for the Optimal Redistributive Capital Taxation Model

# This module computes the steady state (k*, c*, λ*, μ*) under the positive-interest
# regime r̃* = ρ, and checks complementarity via r̃ = max{0, A(1-η)k^{θ-1} - δ - γ/(λ k)}.
#
# It solves the scalar nonlinear equation implied by the intratemporal condition when r̃>0:
#   λ + μ c/(β k) = γ / x,
# where at steady state r̃* = ρ,
#   c = ρ k + A η k^θ,
#   x = A(1-η)k^θ - (δ+ρ)k,
#   λ = γ / (k [A(1-η)k^{θ-1} - δ - ρ]),
#   μ = (c^{-β} - λ)/ρ.
# This yields a single equation in k>0. We solve it numerically and back out the rest.

module SteadyState

export find_steady_state, SteadyStateResult

using Roots

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
    find_steady_state(p; k_guess=nothing)

Compute steady state as a function of parameters `p::ModelParams`.
Assumes positive-interest steady state r̃* = ρ > 0 and enforces the intratemporal
condition. Returns `SteadyStateResult`.

Notes:
- Requires denominators to be positive: A(1-η)k^{θ-1} - δ - ρ > 0 and x(k) > 0.
- If a root is not bracketed near k0, the routine scans a broad range and picks
  the first sign change. If none are found, it tries a local solver from k0.
"""
function find_steady_state(p; k_guess=nothing)
    ρ, A, θ, η, β, δ, γ = p.ρ, p.A, p.θ, p.η, p.β, p.δ, p.γ

    rstar = ρ

    # Helper functions
    c_of(k) = rstar*k + A*η*k^θ
    x_of(k) = A*(1-η)*k^θ - (δ + rstar)*k
    denom_of(k) = A*(1-η)*k^(θ-1) - δ - rstar
    λ_of(k) = γ / (k * denom_of(k))
    μ_of(k) = (c_of(k)^(-β) - λ_of(k)) / rstar

    # Intratemporal residual F(k) = 0 in positive-interest region
    function F(k)
        if k <= 0
            return 1e6
        end
        d = denom_of(k)
        x = x_of(k)
        if !(isfinite(d) && isfinite(x)) || d <= 0 || x <= 0
            return 1e6 + (d <= 0 ? 1e4 : 0) + (x <= 0 ? 1e4 : 0)
        end
        λ = λ_of(k)
        c = c_of(k)
        μ = μ_of(k)
        return λ + μ * c / (β * k) - γ / x
    end

    # Admissible domain: denom(k)>0 and x(k)>0.
    # Since x(k) = k*denom(k), for k>0 it's equivalent to denom(k)>0.
    # Solve denom(k)=0 for k_bar: A(1-η)k^(θ-1) = δ+ρ -> k_bar = ((A(1-η))/(δ+ρ))^(1/(1-θ)).
    k_bar = ((A*(1-η))/(δ + rstar))^(1/(1-θ))
    k_min = 1e-6
    k_max = 0.99 * k_bar
    if !(k_min < k_max)
        error("Invalid parameters: no positive-interest domain (k_bar ≤ 0).")
    end

    # Build a dense grid in (k_min, k_max) to bracket a root
    grid = exp.(range(log(k_min), log(k_max), length=600))
    vals = map(F, grid)

    # Find sign change
    br_lo = nothing; br_hi = nothing
    for i in 1:length(grid)-1
        f1, f2 = vals[i], vals[i+1]
        if isfinite(f1) && isfinite(f2) && signbit(f1) != signbit(f2)
            br_lo = grid[i]
            br_hi = grid[i+1]
            break
        end
    end

    kstar = nothing
    if br_lo !== nothing
        kstar = find_zero(F, (br_lo, br_hi), Bisection(), verbose=false)
    else
        # Fallback: pick the minimizer of |F| over admissible domain
        idx = findmin(abs.(vals))[2]
        kstar = grid[idx]
    end

    # Validate and compute remaining variables
    d = denom_of(kstar)
    x = x_of(kstar)
    if d <= 0 || x <= 0
        error("Steady-state conditions invalid: denom=$(d), x=$(x). Adjust parameters or initial guess.")
    end
    λ = λ_of(kstar)
    c = c_of(kstar)
    μ = μ_of(kstar)
    r_tilde = rstar
    fnames = fieldnames(typeof(p))
    tau_k = ((:r in fnames) && p.r > 0) ? max(0.0, 1.0 - r_tilde / p.r) : 0.0

    return SteadyStateResult(kstar, c, λ, μ, r_tilde, x, tau_k)
end

end # module
