module OptimalWealthTax

using BoundaryValueDiffEq
using DifferentialEquations
using ForwardDiff
using Ipopt
using NLsolve
using Parameters

include("OptimalWealthTax/parameters.jl")
include("OptimalWealthTax/model.jl")
include("OptimalWealthTax/steady_state.jl")
include("OptimalWealthTax/solver.jl")
include("OptimalWealthTax/visualization.jl")

export ModelParams
export SteadyStateResult, find_steady_state
export CollocationResult, solve_collocation, solve_shooting, solve_multiple_shooting
export continue_horizon_stages, solve_collocation_staged_horizon
export refine_with_bvp, compare_solution_paths
export transversality_metrics
export control_reconstruction_diagnostics, boundary_activity_series, boundary_activity_metrics
export production_terms, foc_implied_controls, dynamics
export plot_solution

end # module