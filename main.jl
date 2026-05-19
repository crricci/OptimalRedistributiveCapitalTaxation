using Pkg
Pkg.activate(@__DIR__)

try
    @eval using DifferentialEquations, BoundaryValueDiffEq, Parameters, PyPlot, NLsolve, Ipopt, ForwardDiff, JuMP
catch
    Pkg.instantiate()
    @eval using DifferentialEquations, BoundaryValueDiffEq, Parameters, PyPlot, NLsolve, Ipopt, ForwardDiff, JuMP
end

include(joinpath(@__DIR__, "src", "NoWealthTaxation.jl"))
include(joinpath(@__DIR__, "src", "follower_best_response.jl"))
include(joinpath(@__DIR__, "src", "NoWealthTaxationTools.jl"))
include(joinpath(@__DIR__, "src", "NoWealthTaxationRun.jl"))
include(joinpath(@__DIR__, "src", "OptimalWealthTaxationReducedRun.jl"))

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
    solveReducedOptimalWealthTaxation(; kwargs...)

Public wrapper for the reduced-form `OptimalWealthTaxation` runner.

Input arguments:
- No positional arguments.

Optional parameters:
- `kwargs...`: any keywords accepted by `_solveReducedOptimalWealthTaxation`, such as `output_dir`, `progress`, `model_kwargs`, and continuation controls.

Output:
- Returns a named tuple with reduced-model parameters, the optimized result path, and generated CSV paths.
- Any vector-valued solution components have length `N`.
"""
function solveReducedOptimalWealthTaxation(; kwargs...)
    return _solveReducedOptimalWealthTaxation(; kwargs...)
end

"""
    checkReducedOptimalWealthTaxationFeasibility(control_path; kwargs...)

Public wrapper for feasibility checks on a candidate reduced-form `r_tilde` path.

Input arguments:
- `control_path`: scalar constant path or vector of length `N` containing candidate `r_tilde` values.

Optional parameters:
- `kwargs...`: any keywords accepted by `_checkReducedOptimalWealthTaxationFeasibility`, such as `output_dir`, `progress`, `model_kwargs`, and `filename_prefix`.

Output:
- Returns a named tuple with reduced-model parameters, the evaluated path, a feasibility flag, and generated output paths.
"""
function checkReducedOptimalWealthTaxationFeasibility(control_path; kwargs...)
    return _checkReducedOptimalWealthTaxationFeasibility(control_path; kwargs...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    solveNoWealthTaxation()
end
