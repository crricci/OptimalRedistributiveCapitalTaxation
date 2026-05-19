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

"""
    SolutionResult

Struct storing the output of a `NoWealthTaxation` solve.

Fields:
- `success::Bool`: solver success flag.
- `t`, `k`, `c`, `λ`, `μ`, `r_tilde`, `tau_k`, `λ_tr`, `μ_tr`, `c_tr`: vectors of common length `N = length(t)`.
- `steady::SteadyStateResult`: scalar steady-state reference.
"""
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
Input arguments:
- `res::SolutionResult`: solution with trajectory length `N = length(res.t)`.

Optional parameters:
- `p = ModelParams()`: parameter object used to evaluate residuals.

Output:
- Returns a `Dict{String, Any}` of scalar max and RMS norms.
- All residual arrays used internally have length `N`.
"""
function check_residuals(res::SolutionResult; p=ModelParams())
    @unpack A, θ, η, ρ, β, δ, γ, min_positive = p
    t = res.t; k = res.k; c = res.c; λ = res.λ; μ = res.μ; r̃ = res.r_tilde
    n = length(t)
    ksafe = max.(k, min_positive); csafe = max.(c, min_positive); λsafe = max.(λ, min_positive)
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
    solve_orct(p; T=p.T, N::Int=p.N, debug::Bool=false, progress::Bool=true)

High-level solver for the `NoWealthTaxation` system, combining shooting continuation and a BVP attempt.

Input arguments:
- `p`: parameter object, typically `ModelParams`.

Optional parameters:
- `T = p.T`: scalar solution horizon.
- `N::Int = p.N`: number of saved time points when the IVP path is stored.
- `debug::Bool = false`: print additional diagnostics for initial guesses and derivatives.
- `progress::Bool = true`: print solver progress.

Output:
- Returns a `SolutionResult`.
- All vector-valued fields in the result have common length `N_path`, equal either to the BVP grid length returned by the solver or to the saved IVP path length.
"""
function solve_orct(p; T=p.T, N::Int=p.N, debug::Bool=false, progress::Bool=true)
    steady = SteadyState.find_steady_state(p)
    @unpack A, θ, η, ρ, β, δ, γ = p
    floor = p.min_positive
    progress && println("Solving ORCT 4D system (k,c,λ,μ); T=$(round(T,digits=2)) k0=$(p.k0)")
    progress && println("Target steady state: k*=$(steady.k), c*=$(steady.c), λ*=$(steady.λ)")

    # ODE in (k,c,λ,μ). r_tilde depends on (k,λ,μ,c) via interior expression:
    #   r_int = A(1-η)k^{θ-1} - δ - (βγ)/(λ β k + μ c)
    #   r_tilde = max(r_int, 0)
    function f!(dY, Y, p_local, t)
        k, c, λ, μ = Y
        k = max(k, floor); c = max(c, floor); λ = max(λ, floor)
        denom = λ*β*k + μ*c
        denom = ifelse(isfinite(denom) && denom > floor, denom, floor)
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
        dY[1] = clamp(dk, -p.derivative_clamp, p.derivative_clamp)
        dY[2] = clamp(dc, -p.derivative_clamp, p.derivative_clamp)
        dY[3] = clamp(dλ, -p.derivative_clamp, p.derivative_clamp)
        dY[4] = clamp(dμ, -p.derivative_clamp, p.derivative_clamp)
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
                denom0 = Y0[3]*β*max(Y0[1], floor) + 0.0 # μ(0)=0 so μ*c term zero here
                denom0 = denom0 > floor ? denom0 : floor
                rt0 = A*(1-η)*max(Y0[1], floor)^(θ-1) - δ - (β*γ)/denom0
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
        dt0 = min(p.ivp_dt_initial_cap, Tcur / p.ivp_dt_initial_divisor)
        dtmax = max(p.ivp_dtmax_floor, Tcur / p.ivp_dtmax_divisor)
        if save
            solve(prob, TRBDF2(); abstol=p.ivp_abstol, reltol=p.ivp_reltol, dt=dt0, dtmin=p.ivp_dtmin, dtmax=dtmax, saveat=range(0.0, Tcur, length=N), maxiters=p.ivp_maxiters)
        else
            solve(prob, TRBDF2(); abstol=p.ivp_abstol, reltol=p.ivp_reltol, dt=dt0, dtmin=p.ivp_dtmin, dtmax=dtmax, save_everystep=false, maxiters=p.ivp_maxiters)
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
            F .= p.solver_failure_penalty
            return F
        end
        if sol.retcode != SciMLBase.ReturnCode.Success || isempty(sol.u)
            F .= p.solver_failure_penalty
            return F
        end
        kT, cT, λT, μT = sol.u[end]
        kTp = max(kT, floor); λTp = max(λT, floor)
        denomT = λTp*β*kTp + μT*max(cT, floor)
        denomT = denomT > floor ? denomT : floor
        rT = A*(1-η)*kTp^(θ-1) - δ - (β*γ)/denomT
        F[1] = kT - steady.k
        F[2] = rT - ρ
        return F
    end

    v0_guess = [log(max(p.shooting_initial_c_scale * steady.c, p.shooting_log_floor)), log(max(steady.λ, p.shooting_log_floor))]
    # Debug: check BVP initial guess at t=0
    if debug && !debug_printed_bvp[]
        Yg0 = [p.k0, max(α*steady.c, 1e-6), max(z_star, 1e-6)]
        dYg0 = similar(Yg0)
        try
            f!(dYg0, copy(Yg0), nothing, 0.0)
            rtg0 = A*(1-η)*max(Yg0[1], floor)^(θ-1) - δ - γ*clamp(Yg0[3], floor, p.derivative_clamp)
            @info "Initial BVP guess state and derivative" Yg0 dYg0 rtg0 isfinite_Yg0=all(isfinite, Yg0) isfinite_dYg0=all(isfinite, dYg0)
        catch err
            @warn "Initial BVP derivative evaluation threw" err Yg0
        end
        debug_printed_bvp[] = true
    end
    stage_schedule = T < p.shooting_stage_cutoff ? p.shooting_stage_schedule_short : p.shooting_stage_schedule_long
    Ts = unique(Float64[min(T, t) for t in (stage_schedule..., T)])
    progress && println("→ Shooting continuation in $(length(Ts)) stage(s): ", join(string.(round.(Ts; digits=2)), ", "))
    for Tcur in Ts
        progress && println("  • Stage T=$(round(Tcur,digits=2))")
        local_res!(F, v) = residual_T!(F, v, Tcur)
        # Multi-start around current guess and steady-state-based guess
        seeds = Vector{Vector{Float64}}()
        push!(seeds, copy(v0_guess))
        base = [log(max(steady.c, p.shooting_log_floor)), log(max(steady.λ, p.shooting_log_floor))]
        for ac in p.shooting_seed_multipliers, az in p.shooting_seed_multipliers
            push!(seeds, [log(max(ac * steady.c, p.shooting_log_floor)), log(max(az * max(steady.λ, p.shooting_log_floor), p.shooting_log_floor))])
        end
        progress && println("    shooting seeds=$(length(seeds)) current guess: c0=$(exp(v0_guess[1])) λ0=$(exp(v0_guess[2]))")
        best_v = copy(v0_guess); best_res = Inf
        c_log_floor = log(p.shooting_log_floor)
        c_log_cap = log(p.shooting_c_cap_scale * max(1.0, steady.c))
        λ_log_cap = log(p.shooting_lambda_cap)
        for (seed_idx, vtry) in enumerate(seeds)
            vtry[1] = clamp(vtry[1], c_log_floor, c_log_cap)
            vtry[2] = clamp(vtry[2], c_log_floor, λ_log_cap)
            nls = NLsolve.nlsolve(local_res!, vtry; xtol=p.shooting_xtol, ftol=p.shooting_ftol, method=:trust_region, autodiff=:forward, iterations=p.shooting_iterations, show_trace=false)
            Ftmp = zeros(2);
            local_res!(Ftmp, nls.zero)
            resn = hypot(Ftmp[1], Ftmp[2])
            if debug
                println("    seed #$(seed_idx): converged=$(nls.f_converged || nls.x_converged) residual=$(resn) c0=$(exp(nls.zero[1])) λ0=$(exp(nls.zero[2])) F=$(Tuple(Ftmp))")
            end
            if (nls.f_converged || nls.x_converged) && resn < best_res
                best_res = resn
                best_v = nls.zero
            elseif resn < best_res
                best_res = resn
                best_v = nls.zero
            end
        end
        v0_guess .= p.shooting_guess_blend_old_weight .* v0_guess .+ p.shooting_guess_blend_new_weight .* best_v
        # keep in domain
        v0_guess[1] = clamp(v0_guess[1], c_log_floor, c_log_cap)
        v0_guess[2] = clamp(v0_guess[2], c_log_floor, λ_log_cap)
        stage_F = zeros(2)
        local_res!(stage_F, v0_guess)
        progress && println("    selected stage guess: c0=$(exp(v0_guess[1])) λ0=$(exp(v0_guess[2])) residual=$(hypot(stage_F[1], stage_F[2])) F=$(Tuple(stage_F))")
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
        kTp = max(kT, floor); λTp = max(λT, floor)
        denomT = λTp*β*kTp + μT*max(cT, floor)
        denomT = denomT > floor ? denomT : floor
        rT = A*(1-η)*kTp^(θ-1) - δ - (β*γ)/denomT
        res[1] = ua[1] - p.k0              # k(0)=k0
        res[2] = ua[4] - 0.0               # μ(0)=0
        res[3] = kT - steady.k             # k(T)=k*
        res[4] = (cT/β)*(rT - ρ)           # dc(T)=0 => rT=ρ
        return nothing
    end
    prob_bvp = BVProblem(f!, bc!, guessY, tspan)
    bvp_dt = max(T / p.bvp_dt_divisor, p.bvp_dt_floor)
    progress && println("→ Solving BVP … dt=$(bvp_dt) initial guess end-state=(k=$(steady.k), c=$(steady.c), λ=$(steady.λ), μ=0.0)")
    sol_bvp = solve(prob_bvp, MIRK6(), dt = bvp_dt, abstol=p.bvp_abstol, reltol=p.bvp_reltol)
    progress && println("  BVP retcode=$(sol_bvp.retcode)")
    if sol_bvp.retcode == SciMLBase.ReturnCode.Success
        tt_bvp = Array(sol_bvp.t)
        Y_bvp = reduce(hcat, sol_bvp.u)
        k_bvp = vec(Y_bvp[1, :]); c_bvp = vec(Y_bvp[2, :]); λ_bvp = vec(Y_bvp[3, :]); μ_bvp = vec(Y_bvp[4, :])
        kT_bvp = k_bvp[end]
        cT_bvp = c_bvp[end]
        λT_bvp = max(λ_bvp[end], floor)
        denomT_bvp = λT_bvp * β * max(kT_bvp, floor) + μ_bvp[end] * max(cT_bvp, floor)
        denomT_bvp = denomT_bvp > floor ? denomT_bvp : floor
        rT_bvp = A * (1 - η) * max(kT_bvp, floor)^(θ - 1) - δ - (β * γ) / denomT_bvp
        progress && println("  ✓ BVP converged with terminal state: kT=$(kT_bvp), cT=$(cT_bvp), λT=$(λ_bvp[end]), μT=$(μ_bvp[end]), rT=$(rT_bvp)")
        tt = tt_bvp
        k = k_bvp; c = c_bvp; λ = λ_bvp; μ = μ_bvp
    else
        progress && println("  ↪ BVP failed, falling back to IVP integrate with c0=$(exp(v0_guess[1])) λ0=$(exp(v0_guess[2]))")
        u0_final = [exp(v0_guess[1]), exp(v0_guess[2])]
        sol = integrate(u0_final, T; save=true)
        progress && println("  IVP retcode=$(sol.retcode) saved_steps=$(length(sol.t))")
        tt = Array(sol.t)
        Y = reduce(hcat, sol.u)
        k = vec(Y[1, :]); c = vec(Y[2, :]); λ = vec(Y[3, :]); μ = vec(Y[4, :])
    end
    ksafe = max.(k, floor); csafe = max.(c, floor); λsafe = max.(λ, floor)
    denom_vec = λsafe .* β .* ksafe .+ μ .* csafe
    denom_vec = map(d -> (isfinite(d) && d > floor) ? d : floor, denom_vec)
    r_tilde = A .* (1 .- η) .* ksafe .^ (θ .- 1) .- δ .- (β*γ) ./ denom_vec

    tau_k = similar(k); λ_tr = similar(k); μ_tr = similar(k); c_tr = similar(k)
    for i in eachindex(k)
        denom = A * θ * (1 - η) * ksafe[i]^(θ - 1) - δ
        tau_k[i] = abs(denom) > 1e-12 ? (1.0 - r_tilde[i] / denom) : 0.0
        λ_tr[i] = exp(-ρ*tt[i]) * λ[i] * k[i]
        μ_tr[i] = exp(-ρ*tt[i]) * μ[i] * c[i]
        c_tr[i] = exp(-ρ*tt[i]) * (csafe[i]^(-β)) * k[i]
    end

    # Success checks at terminal time
    kT = k[end]; cT = c[end]; λT = max(λ[end], floor)
    denom_end = λT*β*max(kT, floor) + μ[end]*max(cT, floor)
    denom_end = denom_end > floor ? denom_end : floor
    rT = A*(1-η)*max(kT, floor)^(θ-1) - δ - (β*γ)/denom_end
    dkT = rT*kT + A*η*max(kT, floor)^θ - cT
    dcT = (cT/β) * (rT - ρ)
    ok_r = abs(rT - ρ) < p.terminal_r_tolerance
    ok_dk = abs(dkT) < p.terminal_kdot_tolerance
    ok_dc = abs(dcT) < p.terminal_cdot_tolerance
    ok_k = abs(kT - steady.k) < p.terminal_k_tolerance
    ok_c = abs(cT - steady.c) < p.terminal_c_tolerance
    ok_tvc = (abs(λ_tr[end]) < p.transversality_tolerance) && (abs(μ_tr[end]) < p.transversality_tolerance)
    ok_ctvc = abs(c_tr[end]) < p.transversality_tolerance
    # Transversality quantities (λ_tr(T), μ_tr(T), c_tr(T)) are now diagnostics only
    # and are NOT part of the success gating per user request.
    success = ok_r && ok_dk && ok_dc && ok_k && ok_c
    progress && println("Done (success=$(success))")
    progress && println("Terminal metrics: rT=$(rT), dkT=$(dkT), dcT=$(dcT), λ_tr(T)=$(λ_tr[end]), μ_tr(T)=$(μ_tr[end]), c_tr(T)=$(c_tr[end])")

    return SolutionResult(success, tt, k, c, λ, μ, r_tilde, tau_k, λ_tr, μ_tr, c_tr, steady)
end

end # module
