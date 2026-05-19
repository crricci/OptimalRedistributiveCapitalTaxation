using DelimitedFiles
using LinearAlgebra
using Printf
using Statistics: mean
using Parameters

"""
    compute_residuals(p, sol)

Computes equation-by-equation residual series for a `NoWealthTaxation` solution.

Input arguments:
- `p::NoWealthTaxation.ModelParams`: model parameters.
- `sol::NoWealthTaxation.SolutionResult`: solution whose trajectory fields have common length `N = length(sol.t)`.

Optional parameters:
- None.

Output:
- Returns a named tuple with vector fields `foc_res`, `eq_k`, `eq_c`, `eq_λ`, `eq_μ`, `r_tilde`, and `x`.
- Every returned vector has length `N`.
"""
function compute_residuals(p::NoWealthTaxation.ModelParams, sol::NoWealthTaxation.SolutionResult)
    @unpack A, θ, η, ρ, β, δ, γ = p
    k = sol.k
    c = sol.c
    λ = sol.λ
    μ = sol.μ
    t = sol.t
    n = length(t)
    denom = λ .* β .* k .+ μ .* c
    denom = map(d -> (isfinite(d) && d > 1e-12) ? d : 1e-12, denom)
    r_tilde = A .* (1 .- η) .* k .^ (θ .- 1) .- δ .- (β * γ) ./ denom
    x = A * (1 - η) .* k .^ θ .- (δ .+ r_tilde) .* k

    function dt(v)
        dv = similar(v)
        dv[1] = (v[2] - v[1]) / (t[2] - t[1])
        for i in 2:n-1
            dtm = t[i] - t[i-1]
            dtp = t[i+1] - t[i]
            dv[i] = ((v[i+1] - v[i]) / dtp * dtm + (v[i] - v[i-1]) / dtm * dtp) / (dtm + dtp)
        end
        dv[end] = (v[end] - v[end-1]) / (t[end] - t[end-1])
        dv
    end

    dk = dt(k)
    dc = dt(c)
    dλ = dt(λ)
    dμ = dt(μ)
    foc_res = λ .+ μ .* c ./ (β .* k) .- γ ./ x
    eq_k = dk .- (r_tilde .* k .+ A * η .* k .^ θ .- c)
    eq_c = dc .- (c ./ β) .* (r_tilde .- ρ)
    eq_λ = dλ .- (λ .* (ρ .- r_tilde .- A * θ * η .* k .^ (θ - 1)) .- (γ ./ x) .* (A * θ * (1 - η) .* k .^ (θ - 1) .- δ .- r_tilde))
    eq_μ = dμ .- (μ .* (ρ .- (r_tilde .- ρ) ./ β) .- c .^ (-β) .+ λ)
    return (; foc_res, eq_k, eq_c, eq_λ, eq_μ, r_tilde, x)
end

"""
    summarize_residuals(label, R)

Prints max and mean absolute residual summaries for a residual bundle returned by `compute_residuals`.

Input arguments:
- `label`: label printed in the summary header.
- `R`: named tuple containing residual vectors of common length `N`.

Optional parameters:
- None.

Output:
- Returns `nothing`.
- Consumes the full residual vectors and prints scalar summaries.
"""
function summarize_residuals(label, R)
    f = v -> (@views (maximum(abs.(v)), mean(abs.(v))))
    fkM, fkA = f(R.eq_k)
    fcM, fcA = f(R.eq_c)
    flM, flA = f(R.eq_λ)
    fmM, fmA = f(R.eq_μ)
    ffM, ffA = f(R.foc_res)
    println("Residual summary [$label]:")
    println(@sprintf("  FOC     max=%8.2e  mean=%8.2e", ffM, ffA))
    println(@sprintf("  k-dot   max=%8.2e  mean=%8.2e", fkM, fkA))
    println(@sprintf("  c-dot   max=%8.2e  mean=%8.2e", fcM, fcA))
    println(@sprintf("  λ-dot   max=%8.2e  mean=%8.2e", flM, flA))
    println(@sprintf("  μ-dot   max=%8.2e  mean=%8.2e", fmM, fmA))
end

"""
    test_steady_state_invariance(p; T=50.0)

Runs the legacy solver starting from the analytical steady state to test invariance numerically.

Input arguments:
- `p::NoWealthTaxation.ModelParams`: baseline model parameters.

Optional parameters:
- `T = 50.0`: requested test horizon. This function currently rebuilds parameters using `p.T` in the internal test problem.

Output:
- Returns a `NoWealthTaxation.SolutionResult` on success.
- Returns `nothing` if the solve fails.
"""
function test_steady_state_invariance(p::NoWealthTaxation.ModelParams; T = 50.0)
    ss = NoWealthTaxation.SteadyState.find_steady_state(p)
    pSS = NoWealthTaxation.ModelParams(A = p.A, θ = p.θ, η = p.η, ρ = p.ρ, β = p.β, δ = p.δ, γ = p.γ, k0 = ss.k, T = p.T)
    try
        result = NoWealthTaxation.solve_orct(pSS)
        R = compute_residuals(pSS, result)
        summarize_residuals("steady-state run", R)
        NoWealthTaxation.plot_main_solution(result, "Steady state trajectory", "solution_steady_state.png"; force = true, half = false)
        return result
    catch err
        println("Solver failed from steady state: $(err)")
        return nothing
    end
end

"""
    test_perturbations(p; factors=[0.9, 1.1])

Runs the legacy solver from multiplicative perturbations around the analytical steady state.

Input arguments:
- `p::NoWealthTaxation.ModelParams`: baseline model parameters.

Optional parameters:
- `factors`: vector of scalar multiplicative perturbations applied to steady-state capital.

Output:
- Returns `nothing`.
- For each factor, the underlying solve produces trajectories whose length is determined by the solver.
"""
function test_perturbations(p::NoWealthTaxation.ModelParams; factors = [0.9, 1.1])
    ss = NoWealthTaxation.SteadyState.find_steady_state(p)
    for α in factors
        println("\n--- Perturbation α=$(α) ---")
        pα = NoWealthTaxation.ModelParams(A = p.A, θ = p.θ, η = p.η, ρ = p.ρ, β = p.β, δ = p.δ, γ = p.γ, k0 = α * ss.k, T = p.T)
        try
            res = NoWealthTaxation.solve_orct(pα)
            println("Success=$(res.success) final k=$(res.k[end])")
            R = compute_residuals(pα, res)
            summarize_residuals("α=$(α)", R)
            fn = "solution_perturbation_$(replace(string(α), '.' => '-')).png"
            NoWealthTaxation.plot_main_solution(res, "Perturbation α=$(α)", fn; force = true)
        catch err
            println("Solver failed for perturbation α=$(α): $(err)")
        end
    end
end

"""
    run_gamma_welfare_scan(; k0=2.0, gamma_values=[...], limit=0, outfile="welfare_gamma_scan.csv", progress=true)

Runs the legacy model over a grid of `γ` values and computes discounted welfare for each solve.

Input arguments:
- No positional arguments.

Optional parameters:
- `k0::Float64 = 2.0`: common initial capital used in the scan.
- `gamma_values::Vector{Float64}`: vector of length `G` containing the `γ` values to scan.
- `limit::Int = 0`: if positive, truncate the scan to the first `limit` values.
- `outfile::AbstractString = "welfare_gamma_scan.csv"`: output CSV path.
- `progress::Bool = true`: print scan progress.

Output:
- Returns `outfile`.
- Writes a CSV with two columns and `G_used` data rows.
"""
function run_gamma_welfare_scan(; k0::Float64 = 2.0,
    gamma_values::Vector{Float64} = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
    limit::Int = 0,
    outfile::AbstractString = "welfare_gamma_scan.csv",
    progress::Bool = true)

    if limit > 0
        gamma_values = first(gamma_values, min(limit, length(gamma_values)))
    end

    out = Array{Any}(undef, length(gamma_values) + 1, 2)
    out[1, 1] = "gamma"
    out[1, 2] = "welfare"

    for (i, γ) in enumerate(gamma_values)
        p = NoWealthTaxation.ModelParams(k0 = k0, γ = γ)
        progress && println("[$(i)/$(length(gamma_values))] γ=$(γ): solving …")

        welfare = NaN
        try
            res = NoWealthTaxation.solve_orct(p; progress = false)
            if length(res.t) > 1
                t = res.t
                c = max.(res.c, 1e-12)
                k = max.(res.k, 1e-12)
                r_tilde = res.r_tilde
                x = max.(p.A * (1 - p.η) .* k .^ p.θ .- (p.δ .+ r_tilde) .* k, 1e-12)
                Uc = (c .^ (1.0 - p.β)) ./ (1.0 - p.β)
                Vx = log.(x)
                disc = exp.(-p.ρ .* t)
                integrand = (γ .* Vx .+ Uc) .* disc
                mask = isfinite.(integrand) .& isfinite.(t)
                if count(mask) >= 2
                    tm = t[mask]
                    ym = integrand[mask]
                    w = 0.0
                    @inbounds for j in 1:length(tm)-1
                        dt = tm[j+1] - tm[j]
                        yi = ym[j]
                        yj = ym[j+1]
                        if isfinite(dt)
                            w += 0.5 * (yi + yj) * dt
                        end
                    end
                    welfare = w
                end
            end
        catch err
            progress && println("  ! solver failed: $(err)")
        end

        out[i + 1, 1] = γ
        out[i + 1, 2] = welfare
        progress && println("  → welfare=$(welfare)")
    end

    writedlm(outfile, out, ',')
    println("✓ Saved welfare results to '$(outfile)' (rows=$(size(out, 1) - 1))")
    return outfile
end

"""
    run_gamma_kstar_scan(; gamma_values=[...], outfile="gamma_kstar_scan.csv", progress=true)

Runs the analytical steady-state computation over a grid of `γ` values and stores `k*`.

Input arguments:
- No positional arguments.

Optional parameters:
- `gamma_values::Vector{Float64}`: vector of length `G` containing the `γ` values to scan.
- `outfile::AbstractString = "gamma_kstar_scan.csv"`: output CSV path.
- `progress::Bool = true`: print scan progress.

Output:
- Returns `outfile`.
- Writes a CSV with two columns and `G` data rows.
"""
function run_gamma_kstar_scan(; gamma_values::Vector{Float64} = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0], outfile::AbstractString = "gamma_kstar_scan.csv", progress::Bool = true)
    out = Array{Any}(undef, length(gamma_values) + 1, 2)
    out[1, 1] = "gamma"
    out[1, 2] = "k_star"

    for (i, γ) in enumerate(gamma_values)
        p = NoWealthTaxation.ModelParams(γ = γ)
        progress && println("[$(i)/$(length(gamma_values))] γ=$(γ): computing k* ...")
        k_star = NaN
        try
            ss = NoWealthTaxation.SteadyState.find_steady_state(p)
            k_star = ss.k
        catch err
            progress && println("  ! steady state failed: $(err)")
        end
        out[i + 1, 1] = γ
        out[i + 1, 2] = k_star
        progress && println("  → k*=$(k_star)")
    end

    writedlm(outfile, out, ',')
    println("✓ Saved gamma vs k* results to '$(outfile)' (rows=$(size(out, 1) - 1))")
    return outfile
end

"""
    run_gamma_steadystate_welfare_scan(; gamma_values=[...], outfile="gamma_steadystate_welfare_scan.csv", progress=true)

Computes steady-state welfare over a grid of `γ` values without solving transition dynamics.

Input arguments:
- No positional arguments.

Optional parameters:
- `gamma_values::Vector{Float64}`: vector of length `G` containing the `γ` values to scan.
- `outfile::AbstractString = "gamma_steadystate_welfare_scan.csv"`: output CSV path.
- `progress::Bool = true`: print scan progress.

Output:
- Returns `outfile`.
- Writes a CSV with two columns and `G` data rows.
"""
function run_gamma_steadystate_welfare_scan(; gamma_values::Vector{Float64} = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0], outfile::AbstractString = "gamma_steadystate_welfare_scan.csv", progress::Bool = true)
    out = Array{Any}(undef, length(gamma_values) + 1, 2)
    out[1, 1] = "gamma"
    out[1, 2] = "steady_state_welfare"

    for (i, γ) in enumerate(gamma_values)
        p = NoWealthTaxation.ModelParams(γ = γ)
        progress && println("[$(i)/$(length(gamma_values))] γ=$(γ): computing steady state welfare ...")
        welfare_ss = NaN
        try
            ss = NoWealthTaxation.SteadyState.find_steady_state(p)
            x_star = max(ss.x, 1e-12)
            c_star = max(ss.c, 1e-12)
            log_x_term = γ * log(x_star)
            crra_term = (c_star ^ (1.0 - p.β)) / (1.0 - p.β)
            welfare_ss = (log_x_term + crra_term) / p.ρ
        catch err
            progress && println("  ! steady state welfare calculation failed: $(err)")
        end
        out[i + 1, 1] = γ
        out[i + 1, 2] = welfare_ss
        progress && println("  → steady state welfare=$(welfare_ss)")
    end

    writedlm(outfile, out, ',')
    println("✓ Saved gamma vs steady state welfare results to '$(outfile)' (rows=$(size(out, 1) - 1))")
    return outfile
end

"""
    steady_linearization_eigs_kclm(p=NoWealthTaxation.ModelParams())

Builds the steady-state Jacobian of the legacy `(k, c, λ, μ)` system and reports its eigenvalues.

Input arguments:
- `p::NoWealthTaxation.ModelParams = NoWealthTaxation.ModelParams()`: parameter set.

Optional parameters:
- None.

Output:
- Returns a named tuple with fields `J`, `eigenvalues`, `classification`, and `counts`.
- `J` is a `4 x 4` matrix, `eigenvalues` is a vector of length 4, and the remaining fields are scalar diagnostics or short named tuples.
"""
function steady_linearization_eigs_kclm(p::NoWealthTaxation.ModelParams = NoWealthTaxation.ModelParams())
    ss = NoWealthTaxation.SteadyState.find_steady_state(p)
    k = max(ss.k, 1e-12)
    c = max(ss.c, 1e-12)
    λ = max(ss.λ, 1e-12)
    μ = ss.μ
    A, θ, η, ρ, β, δ, γ = p.A, p.θ, p.η, p.ρ, p.β, p.δ, p.γ

    r_tilde = ρ
    dr_dk = A * (1 - η) * (θ - 1) * k ^ (θ - 2) + γ / (λ * k ^ 2)
    dr_dλ = γ / (λ ^ 2 * k)

    x = A * (1 - η) * k ^ θ - (δ + r_tilde) * k
    dx_dk = A * (1 - η) * θ * k ^ (θ - 1) - (δ + r_tilde) - k * dr_dk
    dx_dλ = -k * dr_dλ

    f1_k = dr_dk * k + r_tilde + A * η * θ * k ^ (θ - 1)
    f1_c = -1.0
    f1_λ = dr_dλ * k
    f1_μ = 0.0

    f2_k = (c / β) * dr_dk
    f2_c = (r_tilde - ρ) / β
    f2_λ = (c / β) * dr_dλ
    f2_μ = 0.0

    S = ρ - r_tilde - A * θ * η * k ^ (θ - 1)
    S_k = -dr_dk - A * θ * η * (θ - 1) * k ^ (θ - 2)
    S_λ = -dr_dλ
    T = A * θ * (1 - η) * k ^ (θ - 1) - δ - r_tilde
    T_k = A * θ * (1 - η) * (θ - 1) * k ^ (θ - 2) - dr_dk
    T_λ = -dr_dλ
    qk = (γ / x) * (T_k - (dx_dk / x) * T)
    qλ = (γ / x) * (T_λ - (dx_dλ / x) * T)
    f3_k = λ * S_k - qk
    f3_c = 0.0
    f3_λ = S + λ * S_λ - qλ
    f3_μ = 0.0

    U = ρ - (r_tilde - ρ) / β
    Uk = -(1 / β) * dr_dk
    Uλ = -(1 / β) * dr_dλ
    g1_k = μ * Uk
    g1_c = β * c ^ (-β - 1)
    g1_λ = μ * Uλ + 1.0
    g1_μ = U

    J = [
        f1_k f1_c f1_λ f1_μ;
        f2_k f2_c f2_λ f2_μ;
        f3_k f3_c f3_λ f3_μ;
        g1_k g1_c g1_λ g1_μ
    ]

    vals = eigvals(J)
    rparts = real.(vals)
    stable = count(x -> x < -1e-9, rparts)
    unstable = count(x -> x > 1e-9, rparts)
    center = length(vals) - stable - unstable
    cls = unstable > 0 && stable > 0 ? "saddle" : (unstable == 0 && stable == length(vals) ? "locally asymptotically stable" : (stable == 0 && unstable > 0 ? "unstable" : "center/degenerate"))

    println("Steady-state linearization (original system; k,c,λ,μ) with γ=$(γ):")
    println("Jacobian J = \n", J)
    println("Eigenvalues = ", vals)
    println("Stability: $(cls)  [stable=$(stable), unstable=$(unstable), center=$(center)]")
    return (J = J, eigenvalues = vals, classification = cls, counts = (stable = stable, unstable = unstable, center = center))
end