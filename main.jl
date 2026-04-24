using Pkg
try
    @eval using DifferentialEquations, BoundaryValueDiffEq, Parameters, PyPlot, NLsolve
catch
    Pkg.activate(".")
    Pkg.add(["DifferentialEquations", "BoundaryValueDiffEq", "Parameters", "PyPlot", "NLsolve"])
    @eval using DifferentialEquations, BoundaryValueDiffEq, Parameters, PyPlot, NLsolve
end

include("NoWealthTaxation.jl")
include("OptimalWealthTax.jl")
include("NoWealthTaxationTools.jl")
include("NoWealthTaxationRun.jl")
include("OptimalWealthTaxationRun.jl")

function solveNoWealthTaxation(; kwargs...)
    return _solveNoWealthTaxation(; kwargs...)
end

function solveOptimalWealthTaxation(; kwargs...)
    return _solveOptimalWealthTaxation(; kwargs...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    solveNoWealthTaxation()
end
