module ORCTSolver

export SolutionResult, solve_orct, check_residuals

using DifferentialEquations
using BoundaryValueDiffEq
using NLsolve
using LinearAlgebra
using Statistics: mean
using Parameters

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
    λ_tr::Vector{Float64}
    μ_tr::Vector{Float64}
    c_tr::Vector{Float64}
    steady::SteadyStateResult
end

"""
    check_residuals(result; p=ModelParams())

Compute residuals for the model equations on the discrete solution path:
  (1) FOC: λ + μ c/(β k) - γ/x = 0
  (2) k̇ - [ r̃ k + A η k^θ - c ] = 0 (finite diff)
  (3) ċ - [ (c/β)(r̃ - ρ) ] = 0
  (4) λ̇ - RHS_λ = 0
  (5) μ̇ - RHS_μ = 0
Return a dictionary with max and RMS norms. Uses central differences for interior points.
"""
function check_residuals(res::SolutionResult; p=ModelParams())
    @unpack A, θ, η, ρ, β, δ, γ = p
    t = res.t; k = res.k; c = res.c; λ = res.λ; μ = res.μ; r̃ = res.r_tilde
    n = length(t)
    ksafe = max.(k,1e-12); csafe = max.(c,1e-12); λsafe = max.(λ,1e-12)
    x = A*(1-η)*ksafe.^θ .- (δ .+ r̃).*ksafe
    foc = λ .+ μ .* csafe ./ (β .* ksafe) .- γ ./ x
    # finite differences
    function fd(y)
        dy = similar(y)
        dy[1] = (y[2]-y[1])/(t[2]-t[1])
        for i in 2:n-1
            dt = t[i+1]-t[i-1]
            dy[i] = (y[i+1]-y[i-1]) / dt
        end
        dy[n] = (y[n]-y[n-1])/(t[n]-t[n-1])
        dy
    end
    kdot = fd(k); cdot = fd(c); λdot = fd(λ); μdot = fd(μ)
    rhs_k = r̃ .* k .+ A*η .* ksafe.^θ .- c
    rhs_c = (csafe ./ β) .* (r̃ .- ρ)
    T = A*θ*(1-η).*ksafe.^(θ-1) .- δ .- r̃
    S = ρ .- r̃ .- A*θ*η.*ksafe.^(θ-1)
    rhs_λ = λ .* S .- (γ ./ x) .* T
    rhs_μ = μ .* ( ρ .- (r̃ .- ρ)./β ) .- csafe.^(-β) .+ λ
    r1 = foc
    r2 = kdot - rhs_k
    r3 = cdot - rhs_c
    r4 = λdot - rhs_λ
    r5 = μdot - rhs_μ
    rms(v) = sqrt(mean(v.^2))
    norms = Dict{String,Any}(
        "FOC_max"=>maximum(abs.(r1)), "FOC_rms"=>rms(r1),
        "k_max"=>maximum(abs.(r2)), "k_rms"=>rms(r2),
        "c_max"=>maximum(abs.(r3)), "c_rms"=>rms(r3),
        "lambda_max"=>maximum(abs.(r4)), "lambda_rms"=>rms(r4),
        "mu_max"=>maximum(abs.(r5)), "mu_rms"=>rms(r5)
    )
    println("Residual norms:")
    for (k,v) in norms
        println(rpad(k,14), "= ", v)
    end
    return norms
end

"""
    solve_orct(p; T=p.T, N::Int=2001, α::Float64=0.95, debug::Bool=false, progress::Bool=true)

2D shooting on (c(0), z(0)) where z ≡ 1/(λ k). Smooth interior ODE in (k,c,z). Targets: k(T)=k*, r̃(T)=ρ.

If `debug=true`, prints the initial conditions and initial derivatives tested for both IVP and BVP guesses and flags any NaN/Inf.
"""
function solve_orct(p; T=p.T, N::Int=2001, debug::Bool=false, progress::Bool=true)
    steady = SteadyState.find_steady_state(p)
    @unpack A, θ, η, ρ, β, δ, γ, r = p
    progress && println("Solving ORCT 4D system (k,c,λ,μ); T=$(round(T,digits=2)) k0=$(p.k0)")

    # ODE in (k,c,λ,μ). r_tilde depends on (k,λ,μ,c) via interior expression:
    #   r_int = A(1-η)k^{θ-1} - δ - (βγ)/(λ β k + μ c)
    #   r_tilde = max(r_int, 0)
    function f!(dY, Y, p_local, t)
        k, c, λ, μ = Y
        k = max(k, 1e-12); c = max(c, 1e-12); λ = max(λ, 1e-12)
        denom = λ*β*k + μ*c
        denom = ifelse(isfinite(denom) && denom > 1e-12, denom, 1e-12)
        r_int = A*(1-η)*k^(θ-1) - δ - (β*γ)/denom
        r_tilde = ifelse(isfinite(r_int), max(r_int, 0.0), 0.0)
        x = A*(1-η)*k^θ - (δ + r_tilde)*k
        if x <= 0
            dY .= 0
            return
        end
        # FOC: λ + μ c/(β k) - γ/x = 0 enforced dynamically by μ equation (not algebraic elimination)
        dk = r_tilde*k + A*η*k^θ - c
        dc = (c/β) * (r_tilde - ρ)
        dλ = λ*(ρ - r_tilde - A*θ*η*k^(θ-1)) - (γ/x)*(A*θ*(1-η)*k^(θ-1) - δ - r_tilde)
        dμ = μ * ( ρ - (r_tilde - ρ)/β ) - c^(-β) + λ
        M = 1e6
        dY[1] = clamp(dk, -M, M)
        dY[2] = clamp(dc, -M, M)
        dY[3] = clamp(dλ, -M, M)
        dY[4] = clamp(dμ, -M, M)
        return nothing
    end

    # one-time debug print guards
    debug_printed_ivp = Ref(false)
    debug_printed_bvp = Ref(false)

    integrate(u0, Tcur; save=false) = begin
        Y0 = [p.k0, u0[1], u0[2], 0.0]  # μ(0)=0
        # Optional initial derivative check to catch NaNs at t=0
        if debug && !debug_printed_ivp[]
            dY0 = similar(Y0)
            try
                f!(dY0, copy(Y0), nothing, 0.0)
                denom0 = Y0[3]*β*max(Y0[1],1e-12) + 0.0 # μ(0)=0 so μ*c term zero here
                denom0 = denom0 > 1e-12 ? denom0 : 1e-12
                rt0 = A*(1-η)*max(Y0[1],1e-12)^(θ-1) - δ - (β*γ)/denom0
                @info "Initial IVP state and derivative" Y0 dY0 rt0 isfinite_Y0=all(isfinite, Y0) isfinite_dY0=all(isfinite, dY0)
            catch err
                @warn "Initial derivative evaluation threw" err Y0
            end
            debug_printed_ivp[] = true
        end
        # Bail out early if initial derivative is non-finite to avoid solver NaN-dt spam
        dY0_chk = similar(Y0)
        f!(dY0_chk, copy(Y0), nothing, 0.0)
        if !(all(isfinite, dY0_chk) && all(isfinite, Y0))
            error("Non-finite initial condition or derivative")
        end
        prob = ODEProblem(f!, Y0, (0.0, Tcur))
        if save
            solve(prob, TRBDF2(); abstol=1e-8, reltol=1e-8, dt=min(1e-3, Tcur/1000), dtmin=1e-12, dtmax=max(1e-2, Tcur/200), saveat=range(0.0, Tcur, length=N), maxiters=20_000_000)
        else
            solve(prob, TRBDF2(); abstol=1e-8, reltol=1e-8, dt=min(1e-3, Tcur/1000), dtmin=1e-12, dtmax=max(1e-2, Tcur/200), save_everystep=false, maxiters=20_000_000)
        end
    end

    # Use steady state values for c, λ as targets; unknown initial c0, λ0 (both positive) -> 2D shooting
    function residual_T!(F, v0, Tcur)
        # v0 are logs to enforce positivity of c0, λ0
        u0 = similar(v0)
        u0[1] = exp(v0[1])  # c0
        u0[2] = exp(v0[2])  # λ0
        local sol
        try
            sol = integrate(u0, Tcur; save=false)
        catch
            F .= 1e6
            return F
        end
        if sol.retcode != SciMLBase.ReturnCode.Success || isempty(sol.u)
            F .= 1e6
            return F
        end
        kT, cT, λT, μT = sol.u[end]
        kTp = max(kT,1e-12); λTp = max(λT,1e-12)
    denomT = λTp*β*kTp + μT*max(cT,1e-12)
    denomT = denomT > 1e-12 ? denomT : 1e-12
    rT = A*(1-η)*kTp^(θ-1) - δ - (β*γ)/denomT
        F[1] = kT - steady.k
        F[2] = rT - ρ
        return F
    end

    v0_guess = [log(max(0.9*steady.c, 1e-6)), log(max(steady.λ, 1e-6))]
    # Debug: check BVP initial guess at t=0
    if debug && !debug_printed_bvp[]
        Yg0 = [p.k0, max(α*steady.c, 1e-6), max(z_star, 1e-6)]
        dYg0 = similar(Yg0)
        try
            f!(dYg0, copy(Yg0), nothing, 0.0)
            rtg0 = A*(1-η)*max(Yg0[1],1e-12)^(θ-1) - δ - γ*clamp(Yg0[3],1e-12,1e6)
            @info "Initial BVP guess state and derivative" Yg0 dYg0 rtg0 isfinite_Yg0=all(isfinite, Yg0) isfinite_dYg0=all(isfinite, dYg0)
        catch err
            @warn "Initial BVP derivative evaluation threw" err Yg0
        end
        debug_printed_bvp[] = true
    end
    Ts = [min(T, t) for t in (T < 120 ? (10.0, 20.0, 40.0, 80.0, T) : (20.0, 40.0, 80.0, 120.0, T))]
    progress && println("→ Shooting continuation in $(length(Ts)) stage(s): ", join(string.(round.(Ts; digits=2)), ", "))
    for Tcur in Ts
        progress && println("  • Stage T=$(round(Tcur,digits=2))")
        local_res!(F, v) = residual_T!(F, v, Tcur)
        # Multi-start around current guess and steady-state-based guess
        seeds = Vector{Vector{Float64}}()
        push!(seeds, copy(v0_guess))
    base = [log(max(steady.c,1e-6)), log(max(steady.λ,1e-6))]
        for ac in (0.8, 1.0, 1.2), az in (0.8, 1.0, 1.2)
            push!(seeds, [log(max(ac*steady.c,1e-8)), log(max(az*max(steady.λ,1e-6), 1e-8))])
        end
        best_v = copy(v0_guess); best_res = Inf
        for vtry in seeds
            vtry[1] = clamp(vtry[1], log(1e-8), log(10*max(1.0, steady.c)))
            vtry[2] = clamp(vtry[2], log(1e-8), log(10.0))
            nls = NLsolve.nlsolve(local_res!, vtry; xtol=1e-10, ftol=1e-10, method=:trust_region, autodiff=:forward, iterations=800, show_trace=false)
            Ftmp = zeros(2);
            local_res!(Ftmp, nls.zero)
            resn = hypot(Ftmp[1], Ftmp[2])
            if (nls.f_converged || nls.x_converged) && resn < best_res
                best_res = resn
                best_v = nls.zero
            elseif resn < best_res
                best_res = resn
                best_v = nls.zero
            end
        end
        v0_guess .= 0.7 .* v0_guess .+ 0.3 .* best_v
        # keep in domain
        v0_guess[1] = clamp(v0_guess[1], log(1e-8), log(10*max(1.0, steady.c)))
        v0_guess[2] = clamp(v0_guess[2], log(1e-8), log(10.0))
    end

    # Try BVP solve (collocation) on (k,c,λ,μ) with stationary landing BCs and μ(0)=0
    tspan = (0.0, T)
    guessY(t) = [
        p.k0 + (steady.k - p.k0)*(t/T),
        steady.c,
        steady.λ,
        0.0
    ]
    # Terminal BCs: k(0)=k0; enforce k(T)=k* and dc(T)=0 (=> r̃(T)=ρ). This targets the correct steady state.
    function bc!(res, u, p_local, t)
        ua = u[1]; ub = u[end]
        kT, cT, λT, μT = ub
        kTp = max(kT, 1e-12); λTp = max(λT,1e-12)
    denomT = λTp*β*kTp + μT*max(cT,1e-12)
    denomT = denomT > 1e-12 ? denomT : 1e-12
    rT = A*(1-η)*kTp^(θ-1) - δ - (β*γ)/denomT
        res[1] = ua[1] - p.k0              # k(0)=k0
        res[2] = ua[4] - 0.0               # μ(0)=0
        res[3] = kT - steady.k             # k(T)=k*
        res[4] = (cT/β)*(rT - ρ)           # dc(T)=0 => rT=ρ
        return nothing
    end
    prob_bvp = BVProblem(f!, bc!, guessY, tspan)
    progress && println("→ Solving BVP …")
    sol_bvp = solve(prob_bvp, MIRK6(), dt = max(T/400, 0.02), abstol=1e-9, reltol=1e-9)
    if sol_bvp.retcode == SciMLBase.ReturnCode.Success
        progress && println("  ✓ BVP converged")
        tt = Array(sol_bvp.t)
        Y = reduce(hcat, sol_bvp.u)
        k = vec(Y[1, :]); c = vec(Y[2, :]); λ = vec(Y[3, :]); μ = vec(Y[4, :])
    else
        progress && println("  ↪ BVP failed, falling back to IVP integrate …")
        u0_final = [exp(v0_guess[1]), exp(v0_guess[2])]
        sol = integrate(u0_final, T; save=true)
        tt = Array(sol.t)
        Y = reduce(hcat, sol.u)
        k = vec(Y[1, :]); c = vec(Y[2, :]); λ = vec(Y[3, :]); μ = vec(Y[4, :])
    end
    ksafe = max.(k, 1e-12); csafe = max.(c, 1e-12); λsafe = max.(λ, 1e-12)
    denom_vec = λsafe .* β .* ksafe .+ μ .* csafe
    denom_vec = map(d -> (isfinite(d) && d > 1e-12) ? d : 1e-12, denom_vec)
    r_tilde = A .* (1 .- η) .* ksafe .^ (θ .- 1) .- δ .- (β*γ) ./ denom_vec

    tau_k = similar(k); λ_tr = similar(k); μ_tr = similar(k); c_tr = similar(k)
    for i in eachindex(k)
        denom = (r - δ)
        tau_k[i] = abs(denom) > 1e-12 ? (1.0 - r_tilde[i] / denom) : 0.0
        λ_tr[i] = exp(-ρ*tt[i]) * λ[i] * k[i]
        μ_tr[i] = exp(-ρ*tt[i]) * μ[i] * c[i]
        c_tr[i] = exp(-ρ*tt[i]) * (csafe[i]^(-β)) * k[i]
    end

    # Success checks at terminal time
    kT = k[end]; cT = c[end]; λT = max(λ[end],1e-12)
    denom_end = λT*β*max(kT,1e-12) + μ[end]*max(cT,1e-12)
    denom_end = denom_end > 1e-12 ? denom_end : 1e-12
    rT = A*(1-η)*max(kT,1e-12)^(θ-1) - δ - (β*γ)/denom_end
    dkT = rT*kT + A*η*max(kT,1e-12)^θ - cT
    dcT = (cT/β) * (rT - ρ)
    ok_r = abs(rT - ρ) < 2e-3
    ok_dk = abs(dkT) < 2e-3
    ok_dc = abs(dcT) < 2e-3
    ok_k = abs(kT - steady.k) < 5e-3
    ok_c = abs(cT - steady.c) < 5e-3
    ok_tvc = (abs(λ_tr[end]) < 1e-2) && (abs(μ_tr[end]) < 1e-2)
    ok_ctvc = abs(c_tr[end]) < 1e-2
    # Transversality quantities (λ_tr(T), μ_tr(T), c_tr(T)) are now diagnostics only
    # and are NOT part of the success gating per user request.
    success = ok_r && ok_dk && ok_dc && ok_k && ok_c
    progress && println("Done (success=$(success))")
    progress && println("Terminal metrics: rT=$(rT), dkT=$(dkT), dcT=$(dcT), λ_tr(T)=$(λ_tr[end]), μ_tr(T)=$(μ_tr[end]), c_tr(T)=$(c_tr[end])")

    return SolutionResult(success, tt, k, c, λ, μ, r_tilde, tau_k, λ_tr, μ_tr, c_tr, steady)
end

end # module
