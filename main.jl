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

const OUTPUT_PNG = "solution.png"
const OUTPUT_PNG_KSTAR = "solution_kstar.png"

function run()
    p = ModelParams()
    println("Parameters: ", p)
    println("Computing steady state…")
    ss = SteadyState.find_steady_state(p)
    println("Steady state:")
    println("  k*=$(ss.k), c*=$(ss.c), λ*=$(ss.λ), μ*=$(ss.μ), r̃*=$(ss.r_tilde), x*=$(ss.x), τ_k*=$(ss.tau_k)")

    println("Solving BVP on [0, $(p.T)]…")
    result = solve_orct(p)
    println("Solver success: ", result.success)

    title = "Optimal path (k0=$(p.k0))"
    plot_main_solution(result, title, OUTPUT_PNG)
    println("Saved plot to $(OUTPUT_PNG)")

    # Rerun with initial condition k(0) = k*
    println("\nSolving with initial capital set to k*…")
    p_kstar = ModelParams(k0 = ss.k, T = max(p.T, 200.0))
    result_kstar = solve_orct(p_kstar)
    println("Solver success (k0=k*): ", result_kstar.success)
    title2 = "Optimal path (k0=k*)"
    # Force plotting so we get a figure even if strict success=false
    plot_main_solution(result_kstar, title2, OUTPUT_PNG_KSTAR; force=true)
    println("Saved plot to $(OUTPUT_PNG_KSTAR)")
end

run()
