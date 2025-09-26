# Optimal Redistributive Capital Taxation (ORCT) – Numerical tools

Julia code to compute the analytical steady state, solve the dynamic ORCT system, assess stability, and run welfare scans across γ. Plots are saved (not shown on screen).

Contents
- `parameters.jl` – parameter struct and defaults
- `steady_state.jl` – (legacy) analytical steady state (\(\tilde r^* = \rho\)) used as an initial reference
- `solver.jl` – 4D ODE in (k, c, λ, μ) with complementarity on \(\tilde r\); shooting + BVP, diagnostics
- `visualization.jl` – save-only plotting (`plot_main_solution`, `plot_welfare_vs_gamma`)
- `main.jl` – runner and utilities (welfare scan, linearizations)

## Current dynamic system (4D with complementarity)

We now solve the full 4–dimensional system in the variables \((k,c,\lambda,\mu)\) with an endogenous effective return \(\tilde r\) subject to a non–negativity (complementarity) condition:

```math
\begin{cases}
\lambda + \dfrac{\mu c}{\beta k} = \dfrac{\gamma}{x},\\[6pt]
\dot{k} = \tilde r\, k + A\eta k^{\theta} - c,\\[4pt]
\dot{c} = \dfrac{c}{\beta}(\tilde r - \rho),\\[6pt]
\dot{\lambda} = \lambda(\rho - \tilde r - A\theta \eta k^{\theta-1}) - \dfrac{\gamma}{x}\Big(A\theta(1-\eta)k^{\theta-1} - \delta - \tilde r\Big),\\[6pt]
\dot{\mu} = \mu\Big( \rho - \dfrac{\tilde r - \rho}{\beta} \Big) - c^{-\beta} + \lambda,\\[6pt]
	\tilde{r} = \max\!\left\{0,\; A(1-\eta)k^{\theta-1} - \delta - \dfrac{\beta\gamma}{\lambda\beta k + \mu c}\right\},\\[10pt]
x = A(1-\eta)k^{\theta} - (\delta + \tilde r)k,\\[4pt]
e^{-\rho t} \lambda(t) k(t) \to 0,\qquad e^{-\rho t} c(t)^{-\beta} k(t) \to 0,\\[4pt]
\mu(0)=0, \qquad k(0)=k_0.
\end{cases}
```

The max–operator encodes the Kuhn–Tucker condition ensuring \(\tilde r \ge 0\) and (on the interior branch) the marginal return is reduced by the redistribution wedge.

### Steady state system (interior branch)

Assuming an interior steady state with \(\tilde r^*>0\) and \(\tilde r^* = \rho\) (since \(\dot c=0\) requires \(\tilde r = \rho\) for \(c^*>0\)) the steady state solves:

```math
\begin{aligned}
0 &= \rho k^* + A\eta (k^*)^{\theta} - c^*,\\
0 &= \lambda^*(\rho - \rho - A\theta\eta (k^*)^{\theta-1}) - \frac{\gamma}{x^*}\Big(A\theta(1-\eta)(k^*)^{\theta-1} - \delta - \rho\Big),\\
0 &= \mu^* \rho - (c^*)^{-\beta} + \lambda^*,\\
\lambda^* + \frac{\mu^* c^*}{\beta k^*} &= \frac{\gamma}{x^*},\\
\rho &= A(1-\eta)(k^*)^{\theta-1} - \delta - \frac{\beta\gamma}{\lambda^*\beta k^* + \mu^* c^*},\\
x^* &= A(1-\eta)(k^*)^{\theta} - (\delta + \rho) k^*.
\end{aligned}
```

These six relations determine \((k^*, c^*, \lambda^*, \mu^*, x^*)\). A legacy analytical closed form (from a pre-revision \(\tilde r\)) is currently used only as an initial reference; a numerical solve matching the present \(\tilde r\) can replace it in future.

### Capital tax rate

Reported along a trajectory as \(\tau_k = 1 - \tilde r/(r-\delta)\) (guarded near \(r=\delta\)).

## Solution strategy

1. **State–costate integration (4D)**: We treat \((k,c,\lambda,\mu)\) as dynamical variables; the algebraic FOC is monitored via residuals (not eliminated) to retain robustness when \(x\) becomes small.
2. **Complementarity**: The effective return candidate \(r_{int} = A(1-\eta)k^{\theta-1} - \delta - (\beta\gamma)/(\lambda\beta k + \mu c)\) is clamped at zero: \(\tilde r = \max(0, r_{int})\).
3. **Shooting continuation**: A 2D shooting over the logarithms of initial \(c_0,\lambda_0\) is performed over an increasing sequence of horizons to approach terminal targets.
4. **Terminal BVP refinement**: A collocation BVP (MIRK6) enforces boundary conditions: \(k(0)=k_0\), \(\mu(0)=0\), \(k(T)=k^*\), and \(\dot c(T)=0\) (equivalently \(\tilde r(T)=\rho\)). If collocation fails, the last shooting IVP trajectory is retained.
5. **Diagnostics & residuals**: Finite–difference residuals are computed for all dynamic equations and the FOC. Transversality proxies \(e^{-\rho t}\lambda k\), \(e^{-\rho t}\mu c\), and \(e^{-\rho t} c^{-\beta} k\) are tracked (diagnostically only).
6. **Success criteria**: Require small terminal deviations in \(\tilde r-\rho\), drifts \(\dot k, \dot c\), and proximity to \(k^*, c^*\). Transversality quantities are reported but not gated.

## Residual diagnostics

After each solve the code reports max and RMS norms of: FOC, capital, consumption, \(\lambda\), and \(\mu\) equations. These help identify if a path is only numerically stabilized (large co-state drift) despite a small FOC residual.

## Welfare functional

For a given \(\gamma\), welfare is approximated by trapezoidal integration of
\[
W(\gamma) = \int_0^T \Big( \gamma \log x(t) + \frac{c(t)^{1-\beta}}{1-\beta} \Big) e^{-\rho t}\,dt,
\]
where \(x = A(1-\eta)k^{\theta} - (\delta + \tilde r) k\).

## Steady state note

The analytical closed form presently in `steady_state.jl` predates the denominator change in \(\tilde r\) and is used only as a legacy approximation / initial reference. A numerical steady state solver matching the current specification is an open enhancement.

## Solver and diagnostics (updated summary)

- Continuation shooting (log-parameterized) → collocation BVP.
- Complementarity via clamped \(\tilde r\).
- Residual & transversality reporting.
- Plotting restricted (optionally) to first half of the horizon for clarity; full-range plots available with `half=false`.

## Running

Plots for two initial conditions (saved only):

```bash
julia --project=. main.jl
```

Output: `solution (k0=2.0).png`, `solution (k0=4.0).png`.

Welfare scan over γ (saves CSV, no plots):

```julia
julia --project=. -e 'include("main.jl"); run_gamma_welfare_scan()'
```

Details: computes an approximation to the welfare integral using trapezoidal quadrature (with \(x = A(1-\eta)k^{\theta} - (\delta + \tilde r) k\)). Use `limit=` to do a subset; results saved to `welfare_gamma_scan.csv`.

Plot welfare vs γ from the saved CSV (square figure, save-only):

```julia
julia --project=. -e 'include("visualization.jl"); plot_welfare_vs_gamma()'
```

Output: `welfare_vs_gamma.png`.

Stability (eigenvalues of linearization):

```julia
julia --project=. -e 'include("main.jl"); steady_linearization_eigs_kclm()'    # (k,c,λ,μ)
```



## Dependencies

- Julia: tested with 1.11
- Packages (direct): DifferentialEquations, BoundaryValueDiffEq, NLsolve, Parameters, PyPlot
- Standard library: DelimitedFiles, LinearAlgebra, Statistics, Pkg

Setup:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()   # resolve and install pinned dependencies
```

If you already had this repo before, after pulling run:
```julia
using Pkg
Pkg.activate(".")
Pkg.resolve(); Pkg.instantiate()
```
