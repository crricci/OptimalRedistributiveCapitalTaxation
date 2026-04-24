module OptimalWealthTax

using NLsolve
using Parameters

include("OptimalWealthTax/parameters.jl")
include("OptimalWealthTax/model.jl")
include("OptimalWealthTax/steady_state.jl")
include("OptimalWealthTax/solver.jl")
include("OptimalWealthTax/visualization.jl")

export ModelParams
export SteadyStateResult, find_steady_state
export CollocationResult, solve_collocation
export production_terms, foc_implied_controls, dynamics
export plot_solution

end # module