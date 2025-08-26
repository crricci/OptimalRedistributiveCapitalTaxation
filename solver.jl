module ORCTSolver

export SolutionResult, solve_orct

using DifferentialEquations
using LinearAlgebra
using Parameters
using NLsolve

include("steady_state.jl")
using .SteadyState

struct SolutionResult
    success::Bool
    t::Vector{Float64}
    k::Vector{Float64}
    c::Vector{Float64}
    λ::Vector{Float64}
    μ::Vector{Float64}
    r_tilde::Vector{Float64}
    tau_k::Vector{Float64}
    λ_transversality::Vector{Float64}
    μ_transversality::Vector{Float64}
    steady::SteadyStateResult
end

"""
    solve_orct(p; T=p.T, N::Int=2000, α_init=0.9)

Solves the OCP's FBS ODE system via a shooting method on [0,T].
Unknown initials [c(0), λ(0), μ(0)] are chosen to satisfy terminal
conditions close to the stable manifold and transversality.
"""
function solve_orct(p; T=p.T, N::Int=2000, α_init=0.9)
    steady = find_steady_state(p)

    @unpack A, θ, η, ρ, β, δ, γ, r = p

    function f!(dY, Y, p_local, t)
        k, c, λ, μ = Y
        k = max(k, 1e-10)
        c = max(c, 1e-10)
        λ = clamp(λ, 1e-12, 1e12)
        # r̃ and x
        denom = A*(1-η)*k^(θ-1) - δ - (γ / (λ * k))
        r_tilde = max(0.0, denom)
        x = A*(1-η)*k^θ - (δ + r_tilde)*k
        x = max(x, 1e-12)
        dY[1] = r_tilde*k + A*η*k^θ - c
        dY[2] = (c/β) * (r_tilde - ρ)
        dY[3] = λ*(ρ - r_tilde - A*θ*η*k^(θ-1)) - (γ/x)*(A*θ*(1-η)*k^(θ-1) - δ - r_tilde)
        dY[4] = μ * (ρ - (r_tilde - ρ)/β) - c^(-β) + λ
        return nothing
    end

    function shoot(u0)
        # Initial state vector
        Y0 = [p.k0, u0[1], u0[2], u0[3]]
        tspan = (0.0, T)
        prob = ODEProblem(f!, Y0, tspan)
        sol = solve(prob, Rosenbrock23(), abstol=1e-8, reltol=1e-8, saveat=range(0, T, length=N), maxiters=1_000_000)
        return sol
    end

    function residual!(F, u0)
        sol = shoot(u0)
        YT = sol.u[end]
        kT, cT, λT, μT = YT
        # r̃(T) and x(T)
        kTc = max(kT, 1e-10); cTc = max(cT, 1e-10); λTc = max(λT, 1e-10)
        rT = max(0.0, A*(1-η)*kTc^(θ-1) - δ - γ/(λTc*kTc))
        xT = A*(1-η)*kTc^θ - (δ + rT)*kTc
        # Residuals: hit steady k, intratemporal/corner, and transversality on λ and μ
        F[1] = kT - steady.k
        F[2] = (rT > 1e-8 && xT > 0) ? ((λT + μT * cTc/(β*kTc)) - (γ / xT)) : rT
        F[3] = exp(-ρ*T) * λT * kT + exp(-ρ*T) * μT * cT  # combine discounted TVCs
        return F
    end

    # Initial guess near steady state
    u0_guess = [α_init*steady.c, steady.λ, steady.μ]
    nls = nlsolve(residual!, u0_guess; method=:trust_region, xtol=1e-9, ftol=1e-9)
    success = nls.f_converged || nls.x_converged
    u0_star = success ? nls.zero : u0_guess

    sol = shoot(u0_star)
    tt = Array(sol.t)
    Y = reduce(hcat, sol.u)
    k = vec(Y[1, :]); c = vec(Y[2, :]); λ = vec(Y[3, :]); μ = vec(Y[4, :])

    r_tilde = similar(k); tau_k = similar(k); λ_tr = similar(k); μ_tr = similar(k)
    for i in eachindex(k)
        ki = max(k[i], 1e-12)
        λi = max(λ[i], 1e-12)
        denom = A*(1-η)*ki^(θ-1) - δ - γ/(λi*ki)
        r_tilde[i] = max(0.0, denom)
        tau_k[i] = r > 0 ? max(0.0, 1.0 - r_tilde[i]/r) : 0.0
        λ_tr[i] = exp(-ρ*tt[i]) * λ[i] * k[i]
        μ_tr[i] = exp(-ρ*tt[i]) * μ[i] * c[i]
    end

    return SolutionResult(success, tt, k, c, λ, μ, r_tilde, tau_k, λ_tr, μ_tr, steady)
end

end # module
