# Optimal Redistributive Capital Taxation Model

This Julia project implements a dynamic economic model for optimal redistributive capital taxation, featuring robust ODE solvers, advanced optimization routines, and comprehensive sensitivity and welfare analysis.

## Features
- **Full ODE solution** up to time horizon `T` (default: 1000)
- **All analysis and plots** use only the first half of the time horizon (`T/2`)
- **Welfare analysis**: Computes the welfare integral up to `T/2` for all gamma values, but the plot title uses the mathematical notation $\int_0^\infty$
- **Gamma sensitivity analysis**: Extracts economic variables at `T/2` for each gamma
- **Optimizer choice**: Supports both Optim.jl (default) and NLopt.jl (with many algorithms)
- **Modern 8-panel visualization**: Plots all key variables for $t \in [0, T/2]$

## Usage
1. **Main solution**: `julia main.jl`  
   - Solves the model and saves the 8-panel plot for $t \in [0, T/2]$
2. **Welfare analysis**: `julia -e "include(\"welfare_analysis.jl\"); run_welfare_analysis()"`
   - Computes and plots welfare as a function of gamma (integral up to `T/2`, plot title shows $\infty$)
3. **Gamma sensitivity**: `julia -e "include(\"gamma_sensitivity.jl\"); gamma_sensitivity_state()"`
   - Plots tax rate and interest rate at `T/2` for a range of gamma values

## Output Files
- `normal_precision_solution.png`: 8-panel solution plot ($t \in [0, T/2]$)
- `welfare_analysis_gamma.png`: Welfare vs gamma plot (title: $\int_0^\infty$)
- `gamma_sensitivity_state.png`: Tax rate & interest rate vs gamma at `T/2`
- `welfare_analysis_results.csv`: Detailed welfare data

## Parameters
Edit `parameters.jl` to control model parameters, optimizer, and output file suffixes.

## Notes
- All ODE solutions are computed up to `T`, but all analysis and plots use only the first half (`T/2`).
- The welfare plot uses the $\infty$ symbol in the title for mathematical clarity, but the integral is computed up to `T/2`.

## Requirements
- Julia 1.6+
- Packages: DifferentialEquations.jl, Optim.jl, NLopt.jl, ForwardDiff.jl, PyPlot, Parameters.jl, Statistics, DelimitedFiles

## License
MIT
