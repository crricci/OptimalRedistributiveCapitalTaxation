module NoWealthTaxation

include("NoWealthTaxation/parameters.jl")
include("NoWealthTaxation/solver.jl")
include("NoWealthTaxation/visualization.jl")

const SteadyState = ORCTSolver.SteadyState

using .ORCTSolver: SolutionResult, solve_orct, check_residuals

export ModelParams
export SteadyState
export SolutionResult, solve_orct, check_residuals
export plot_main_solution, plot_welfare_vs_gamma, plot_gamma_vs_kstar, plot_gamma_vs_steadystate_welfare

end # module