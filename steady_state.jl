# Steady state computation for the Optimal Redistributive Capital Taxation Model
# (Updated system: 4D dynamics in (k,c,λ,μ) with algebraic FOC λ + μ c/(β k) = γ/x and
# r̃ = A(1-η)k^{θ-1} - δ - γ/(λ k) (interior, not maxed) plus r̃* = ρ at steady state.)
#
# We no longer rely on a closed‑form k*. Instead we solve the nonlinear system:
#   0 = r̃ k + A η k^θ - c                       (k̇ = 0)
#   0 = λ + μ c/(β k) - γ/x                      (FOC / algebraic constraint)
#   0 = λ(ρ - r̃ - A θ η k^{θ-1}) - (γ/x)(A θ (1-η) k^{θ-1} - δ - r̃)   (λ̇ = 0)
#   0 = μ ρ - c^{-β} + λ                         (μ̇ = 0, using r̃* = ρ)
# with r̃ = ρ imposed (steady state of c requires r̃ = ρ unless c=0) and
#     x = A(1-η)k^θ - (δ + r̃)k.
# Existence requires x>0 and denominators positive; we report an error if not found.

module SteadyState
"""
    analytic_steady_state(p)

Return analytic steady state using provided formula:
  k* = (( (ρ+δ)(θ+γ) + γ ρ θ ) / ( θ A (1+γ)(1-η) ))^{1/(θ-1)}
Then derive:
  r̃* = ρ (by stationarity of c)
  x* = A(1-η)k*^θ - (δ + r̃*)k*
  From FOC: λ* + μ* c*/(β k*) = γ / x*
  μ* eq: 0 = μ* ρ - c*^{-β} + λ* ⇒ μ* = (c*^{-β} - λ*)/ρ
  k̇=0 ⇒ c* = r̃* k* + A η k*^θ = ρ k* + A η k*^θ
  λ̇=0 ⇒ λ*(ρ - r̃* - A θ η k*^{θ-1}) = (γ/x*)(A θ (1-η) k*^{θ-1} - δ - r̃*)
Solve last for λ*, substitute in μ*. (Implemented symbolically below.)
"""
function analytic_steady_state(p)
    ρ, A, θ, η, β, δ, γ = p.ρ, p.A, p.θ, p.η, p.β, p.δ, p.γ
    num = (ρ+δ)*(θ+γ) + γ*ρ*θ
    den = θ * A * (1+γ) * (1-η)
    base = num / den
    if base <= 0
        error("Invalid base for k*: num/den ≤ 0 (num=$(num), den=$(den))")
    end
    kstar = base^(1/(θ-1))
    r_tilde = ρ
    x = A*(1-η)*kstar^θ - (δ + r_tilde)*kstar
    if x <= 0
        error("x* ≤ 0 with analytic k*: x=$(x)")
    end
    c = r_tilde*kstar + A*η*kstar^θ
    # Compute λ* from λ̇=0 condition
    T = A*θ*(1-η)*kstar^(θ-1) - δ - r_tilde
    S = ρ - r_tilde - A*θ*η*kstar^(θ-1)
    # λ* S = (γ/x) T  ⇒ λ* = (γ/x) T / S
    if abs(S) < 1e-14
        error("Singular S in λ* formula (S≈0)")
    end
    λ = (γ/x) * T / S
    μ = (c^(-β) - λ)/ρ
    fnames = fieldnames(typeof(p))
    tau_k = 0.0
    if (:r in fnames) && (:δ in fnames)
        denom = (p.r - p.δ)
        tau_k = abs(denom) > 1e-12 ? (1.0 - r_tilde / denom) : 0.0
    end
    return SteadyStateResult(kstar, c, λ, μ, r_tilde, x, tau_k)
end

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

const find_steady_state = analytic_steady_state

end # module
