# Runner script to compute steady state, solve the BVP, and plot

using Pkg
try
    @eval using DifferentialEquations, BoundaryValueDiffEq, Roots, Parameters, PyPlot
catch
    # Install needed packages if missing
    Pkg.activate(".")
    Pkg.add(["DifferentialEquations", "Roots", "Parameters", "PyPlot", "NLsolve"])
    @eval using DifferentialEquations, Roots, Parameters, PyPlot, NLsolve
end

include("parameters.jl")
include("steady_state.jl")
include("solver.jl")
include("visualization.jl")

using .SteadyState
using .ORCTSolver

const OUTPUT_PNG = "solution.png"

function run()
    p = ModelParams()
    println("Parameters: ", p)
    println("Computing steady state…")
    ss = find_steady_state(p)
    println("Steady state:")
    println("  k*=$(ss.k), c*=$(ss.c), λ*=$(ss.λ), μ*=$(ss.μ), r̃*=$(ss.r_tilde), x*=$(ss.x), τ_k*=$(ss.tau_k)")

    println("Solving BVP on [0, $(p.T)]…")
    result = solve_orct(p)
    println("Solver success: ", result.success)

    title = "Optimal path (k0=$(p.k0))"
    plot_main_solution(result, title, OUTPUT_PNG)
    println("Saved plot to $(OUTPUT_PNG)")
end

run()
