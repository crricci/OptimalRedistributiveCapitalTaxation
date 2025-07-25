# Optimal Redistributive Capital Taxation Model

This Julia project implements a dynamic economic model for optimal redistributive capital taxation, featuring robust ODE solvers, advanced optimization routines, and comprehensive sensitivity and welfare analysis.

## System of Equations

The model solves a system of four ordinary differential equations (ODEs) with two state variables (capital $k$, consumption $c$) and two costate variables ($\lambda$, $\mu$):

- $k' = \tilde{r}k + A\eta k^\theta - c$
- $c' = \frac{c}{\beta}(\tilde{r} - \rho)$
- $\lambda' = \lambda(\rho - \tilde{r} - A\theta\eta k^{\theta-1}) - \frac{\gamma}{x}(A\theta(1-\eta)k^{\theta-1} - \delta - \tilde{r})$
- $\mu' = \mu\left(\rho - \frac{\tilde{r} - \rho}{\beta}\right)$

where $\tilde{r}$ and $x$ are auxiliary variables determined by the model’s constraints.

## Boundary Conditions

- Initial conditions: $k(0) = k_0$ (given), $c(0)$, $\lambda(0)$, and $\mu(0)$ (to be determined).
- Terminal (transversality) conditions at $t = T$:
  - $e^{-\rho T} \lambda(T) k(T) \to 0$
  - $e^{-\rho T} \mu(T) c(T) \to 0$
  - $e^{-\rho T} c(T)^{-\beta} k(T) \to 0$

## Shooting Method

To solve this boundary value problem, a shooting method is used:
- The initial values for $c(0)$, $\lambda(0)$, and $\mu(0)$ are treated as unknowns.
- For a given guess of these initial values, the ODE system is integrated forward in time from $t=0$ to $t=T$.
- At $t=T$, the transversality (terminal) conditions are evaluated.
- An optimizer (either `Optim.jl` or `NLopt.jl`) iteratively adjusts the initial guesses to minimize the sum of the absolute values of the transversality errors at $t=T$.
- The process continues until the terminal conditions are satisfied to a specified tolerance, yielding the unique solution consistent with both the initial and terminal conditions.

## Numerical Solution Method

- The ODE system is solved using the `DifferentialEquations.jl` package with a stiff solver (`RadauIIA5()`).
- Initial conditions for $c$, $\lambda$, and $\mu$ are determined via a shooting method, using either `Optim.jl` (BFGS) or `NLopt.jl` (various algorithms).
- The shooting method minimizes the sum of transversality errors at the final time, ensuring the solution satisfies the model’s boundary conditions.
- Multiple starting points are tried for robustness, and both gradient-based and derivative-free optimizers are supported.


## Features
- **Full ODE solution** up to time horizon `T` (default: 1000)
- **Welfare analysis**: Computes the welfare integral for all gamma values (the plot title uses the mathematical notation $\int_0^\infty$)
- **Gamma sensitivity analysis**: Extracts economic variables at a range of gamma values
- **Optimizer choice**: Supports both Optim.jl (default) and NLopt.jl (with many algorithms)
- **Modern 8-panel visualization**: Plots all key variables for $t \in [0, T]$


## Usage
1. **Main solution**: `julia main.jl`  
   - Solves the model and saves the 8-panel plot for $t \in [0, T]$
2. **Welfare analysis**: `julia -e "include(\"welfare_analysis.jl\"); run_welfare_analysis()"`
   - Computes and plots welfare as a function of gamma (plot title shows $\infty$)
3. **Gamma sensitivity**: `julia -e "include(\"gamma_sensitivity.jl\"); gamma_sensitivity_state()"`
   - Plots tax rate and interest rate for a range of gamma values


## Output Files
- `normal_precision_solution.png`: 8-panel solution plot ($t \in [0, T]$)
- `welfare_analysis_gamma.png`: Welfare vs gamma plot (title: $\int_0^\infty$)
- `gamma_sensitivity_state.png`: Tax rate & interest rate vs gamma
- `welfare_analysis_results.csv`: Detailed welfare data

## Parameters
Edit `parameters.jl` to control model parameters, optimizer, and output file suffixes.



## Requirements
- Julia 1.6+
- Packages: DifferentialEquations.jl, Optim.jl, NLopt.jl, ForwardDiff.jl, PyPlot, Parameters.jl, Statistics, DelimitedFiles

## License
MIT
