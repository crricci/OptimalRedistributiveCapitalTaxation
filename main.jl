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


run();
