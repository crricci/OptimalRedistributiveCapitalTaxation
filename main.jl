# Runner script to compute steady state, solve the BVP, and plot

using Pkg
try
    @eval using DifferentialEquations, BoundaryValueDiffEq, Roots, Parameters, PyPlot, NLsolve
catch
    # Install needed packages if missing
    Pkg.activate(".")
    Pkg.add(["DifferentialEquations", "BoundaryValueDiffEq", "Roots", "Parameters", "PyPlot", "NLsolve"])
    @eval using DifferentialEquations, BoundaryValueDiffEq, Roots, Parameters, PyPlot, NLsolve
end

include("parameters.jl")
include("steady_state.jl")
include("solver.jl")
include("visualization.jl")

using .SteadyState
using .ORCTSolver
using DelimitedFiles
using LinearAlgebra


function run()
    # k0 = 2.0
    p = ModelParams(k0 = 2.0)
    println("Parameters: ", p)
    println("Computing steady state…")
    ss = SteadyState.find_steady_state(p)
    println("Steady state:")
    println("  k*=$(ss.k), c*=$(ss.c), λ*=$(ss.λ), μ*=$(ss.μ), r̃*=$(ss.r_tilde), x*=$(ss.x), τ_k*=$(ss.tau_k)")

    println("Solving BVP on [0, $(p.T)]…")
    result = solve_orct(p)
    println("Solver success: ", result.success)
    title = "Optimal path (k0=$(p.k0))"
    plot_main_solution(result, title, "solution (k0=$(p.k0)).png")
    println("Saved plot")


    # k0 = 4.0
    p = ModelParams(k0 = 4.0)
    println("Parameters: ", p)
    println("Computing steady state…")
    ss = SteadyState.find_steady_state(p)
    println("Steady state:")
    println("  k*=$(ss.k), c*=$(ss.c), λ*=$(ss.λ), μ*=$(ss.μ), r̃*=$(ss.r_tilde), x*=$(ss.x), τ_k*=$(ss.tau_k)")

    println("Solving BVP on [0, $(p.T)]…")
    result = solve_orct(p)
    println("Solver success: ", result.success)
    title = "Optimal path (k0=$(p.k0))"
    plot_main_solution(result, title, "solution (k0=$(p.k0)).png")
    println("Saved plot")

end

# Compute welfare over a range of γ values (no plotting). Saves CSV with columns: gamma,welfare
function run_gamma_welfare_scan(; k0::Float64=2.0,
    gamma_values::Vector{Float64} = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
    limit::Int=0,
    outfile::AbstractString="welfare_gamma_scan.csv",
    progress::Bool=true)

    if limit > 0
        gamma_values = first(gamma_values, min(limit, length(gamma_values)))
    end

    # Prepare output matrix with header
    out = Array{Any}(undef, length(gamma_values) + 1, 2)
    out[1,1] = "gamma"; out[1,2] = "welfare"

    for (i, γ) in enumerate(gamma_values)
        p = ModelParams(k0=k0, γ=γ)
        progress && println("[$(i)/$(length(gamma_values))] γ=$(γ): solving …")
        res = solve_orct(p; progress=false)

        welfare = NaN
        if res.success && length(res.t) > 1
            t = res.t
            c = max.(res.c, 1e-12)
            λ = max.(res.λ, 1e-12)
            # x = γ / λ (since x = γ z k and λ = 1/(z k))
            x = γ ./ λ
            if all(x .> 0)
                # U(c) = c^(1-β)/(1-β); V(x)=log(x)
                Uc = (c .^ (1.0 - p.β)) ./ (1.0 - p.β)
                Vx = log.(x)
                disc = exp.(-p.ρ .* t)
                integrand = (γ .* Vx .+ Uc) .* disc
                # Trapezoidal integration over t
                w = 0.0
                for j in 1:length(t)-1
                    dt = t[j+1] - t[j]
                    yi = integrand[j]
                    yj = integrand[j+1]
                    w += 0.5 * (yi + yj) * dt
                end
                welfare = w
            end
        end
        out[i+1, 1] = γ
        out[i+1, 2] = welfare
        progress && println("  → welfare=$(welfare)")
    end

    writedlm(outfile, out, ',')
    println("✓ Saved welfare results to '$(outfile)' (rows=$(size(out,1)-1))")
    return outfile
end


# Only auto-run the demo when executing this file as a script
if abspath(PROGRAM_FILE) == @__FILE__
    run()
end


# Compute eigenvalues in variables (k, c, λ, μ) at the steady state
function steady_linearization_eigs_kclm(p::ModelParams=ModelParams())
    ss = SteadyState.find_steady_state(p)
    k = max(ss.k, 1e-12)
    c = max(ss.c, 1e-12)
    λ = max(ss.λ, 1e-12)
    A, θ, η, ρ, β, δ, γ = p.A, p.θ, p.η, p.ρ, p.β, p.δ, p.γ

    # r̃ = A(1-η)k^(θ-1) - δ - γ/(λ k)
    r_tilde = ρ  # at steady state
    dr_dk = A*(1-η)*(θ-1)*k^(θ-2) + γ/(λ*k^2)
    dr_dλ = γ/(λ^2 * k)

    # k̇ = A k^θ - δ k - γ/λ - c
    f1_k = A*θ*k^(θ-1) - δ
    f1_c = -1.0
    f1_λ = γ / λ^2

    # ċ = (c/β)(r̃ - ρ)
    f2_k = (c/β) * dr_dk
    f2_c = (r_tilde - ρ) / β  # = 0 at steady state
    f2_λ = (c/β) * dr_dλ

    # λ̇ = λ (ρ + δ - A θ k^(θ-1))
    f3_k = - λ * A * θ * (θ-1) * k^(θ-2)
    f3_c = 0.0
    f3_λ = (ρ + δ - A*θ*k^(θ-1))

    # μ̇ = ( -β c^{-β-1} ċ - λ̇ ) / ρ
    g1_k = ( - (c^(-β)) * dr_dk + λ * A * θ * (θ-1) * k^(θ-2) ) / ρ
    g1_c = 0.0
    g1_λ = ( - c^(-β) * dr_dλ - (ρ + δ - A*θ*k^(θ-1)) ) / ρ
    g1_μ = 0.0

    J = [
        f1_k  f1_c  f1_λ  0.0;
        f2_k  f2_c  f2_λ  0.0;
        f3_k  f3_c  f3_λ  0.0;
        g1_k  g1_c  g1_λ  g1_μ
    ]

    vals = eigvals(J)
    rparts = real.(vals)
    stable = count(x -> x < -1e-9, rparts)
    unstable = count(x -> x > 1e-9, rparts)
    center = length(vals) - stable - unstable
    cls = unstable > 0 && stable > 0 ? "saddle" :
          (unstable == 0 && stable == length(vals) ? "locally asymptotically stable" :
          (stable == 0 && unstable > 0 ? "unstable" : "center/degenerate"))

    println("Steady-state linearization (k,c,λ,μ) with γ=$(γ):")
    println("Jacobian J = \n", J)
    println("Eigenvalues = ", vals)
    println("Stability: $(cls)  [stable=$(stable), unstable=$(unstable), center=$(center)]")
    println("Note: μ is algebraically linked to (c,λ); expect a center (zero) eigenvalue. Dynamics are governed by (k,c,λ).")
    return (J=J, eigenvalues=vals, classification=cls, counts=(stable=stable, unstable=unstable, center=center))
end
