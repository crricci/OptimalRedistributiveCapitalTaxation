using Pkg
Pkg.activate(@__DIR__)

try
    @eval using DifferentialEquations, BoundaryValueDiffEq, Parameters, PyPlot, NLsolve
catch
    Pkg.instantiate()
    @eval using DifferentialEquations, BoundaryValueDiffEq, Parameters, PyPlot, NLsolve
end

include(joinpath(@__DIR__, "src", "NoWealthTaxation.jl"))
include(joinpath(@__DIR__, "src", "OptimalWealthTax.jl"))
include(joinpath(@__DIR__, "src", "NoWealthTaxationTools.jl"))
include(joinpath(@__DIR__, "src", "NoWealthTaxationRun.jl"))
include(joinpath(@__DIR__, "src", "OptimalWealthTaxationRun.jl"))

"""
    solveNoWealthTaxation(; kwargs...)

Public wrapper for the `NoWealthTaxation` runner.

Input arguments:
- No positional arguments.

Optional parameters:
- `kwargs...`: any keyword accepted by `_solveNoWealthTaxation`, such as `k0_values`, `output_dir`, `model_kwargs`, `solve_kwargs`, and `run_gamma_scans`.

Output:
- Returns a named tuple containing runner results, saved file paths, and optional `γ`-scan outputs.
- Any numerical trajectories stored inside the result have length equal to the number of time nodes produced by the underlying solver.
"""
function solveNoWealthTaxation(; kwargs...)
    return _solveNoWealthTaxation(; kwargs...)
end

"""
    solveOptimalWealthTaxation(; kwargs...)

Public wrapper for the `OptimalWealthTax` runner.

Input arguments:
- No positional arguments.

Optional parameters:
- `kwargs...`: any keyword accepted by `_solveOptimalWealthTaxation`, such as `output_dir`, `progress`, `model_kwargs`, and `solve_kwargs`.

Output:
- Returns a named tuple with effective parameters, solve options, the collocation solution, and the generated CSV/PNG paths.
- The trajectories in the `result` field have length `N`, where `N` is the number of collocation grid nodes used in the run.
"""
function solveOptimalWealthTaxation(; kwargs...)
    return _solveOptimalWealthTaxation(; kwargs...)
end

"""
    x(; kwargs...)

Legacy alias of `solveOptimalWealthTaxation`, kept for backward compatibility with older scripts.

Input arguments:
- No positional arguments.

Optional parameters:
- `kwargs...`: same keywords supported by `solveOptimalWealthTaxation`.

Output:
- Returns the same named tuple as `solveOptimalWealthTaxation`.
- Any vector-valued solution components have length `N`.
"""
function x(; kwargs...)
    return _solveOptimalWealthTaxation(; kwargs...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    solveNoWealthTaxation()
end
