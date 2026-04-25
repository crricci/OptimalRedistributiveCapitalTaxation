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

function solveNoWealthTaxation(; kwargs...)
    return _solveNoWealthTaxation(; kwargs...)
end

function solveOptimalWealthTaxation(; kwargs...)
    return _solveOptimalWealthTaxation(; kwargs...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    solveNoWealthTaxation()
end
