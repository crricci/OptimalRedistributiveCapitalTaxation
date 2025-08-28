# Optimal Redistributive Capital Taxation (ORCT) – Numerical tools

Julia code to compute the analytical steady state, solve the dynamic ORCT system, assess stability, and run welfare scans across γ. Plots are saved (not shown on screen).

Contents
- `parameters.jl` – parameter struct and defaults
- `steady_state.jl` – analytical steady state (r̃* = ρ) and derived values
- `solver.jl` – interior ODE in (k, c, z) with z ≡ 1/(λ k), BVP collocation + IVP fallback, diagnostics
- `visualization.jl` – save-only plotting (`plot_main_solution`, `plot_welfare_vs_gamma`)
- `main.jl` – runner and utilities (welfare scan, linearizations)

## Original model

```math
\begin{cases}
\lambda + \frac{\mu c}{\beta k} = \frac{\gamma}{x} & \text{for } \tilde{r} > 0 \\
\dot{k} = \tilde{r} k + A \eta k^{\theta} - c & \text{State eq. for capital} \\
\dot{c} = \frac{c}{\beta} (\tilde{r} - \rho) & \text{State eq. for consumption} \\
\dot{\lambda} = \lambda(\rho - \tilde{r} - A \theta \eta k^{\theta-1}) - \frac{\gamma}{x}\left(A \theta (1-\eta) k^{\theta-1} - \delta - \tilde{r}\right) & \text{Co-state eq. for } \lambda \\
\dot{\mu} = \mu \left( \rho - \frac{\tilde{r} - \rho}{\beta} \right) - c^{-\beta} + \lambda & \text{Co-state eq. for } \mu \\
\lim_{t \to \infty} e^{-\rho t} c(t)^{-\beta}k(t) = 0\\
x = A(1-\eta)k^{\theta} - (\delta + \tilde{r})k & \text{Definition of } x \\
\lim_{t \to \infty} e^{-\rho t} \lambda(t) k(t) = 0 \quad \text{and} \quad \lim_{t \to \infty} e^{-\rho t} \mu(t) c(t) = 0 & \text{Transversality conditions}\\
\tilde{r} = \max \left\{ 0,  A(1-\eta)k^{\theta-1} - \delta - \frac{\gamma}{\lambda k} \right\}
\end{cases}
```
## Model summary

States: (k, c, z) with z = 1/(λ k). Derived: λ = 1/(z k), μ = (c^{−β} − λ)/ρ, x = γ z k.

Derivation of μ(c,λ) used numerically:
- Current-value co-state for μ is μ̇ = μ[ρ − (r̃ − ρ)/β] − c^{−β} + λ.
- Define the homogeneous part μ̇_h = μ_h[ρ − (r̃ − ρ)/β]; the TVC e^{−ρ t} μ c → 0 removes the explosive homogeneous solution.
- On the admissible (non-explosive) branch, the pointwise balance reduces to the particular solution μ = (c^{−β} − λ)/ρ, which we enforce when reconstructing μ along the path while solving only for (k,c,z).

Effective return and ODEs (interior path):
- r̃ = A(1−η) k^{θ−1} − δ − γ z
- k̇ = r̃ k + A η k^θ − c
- ċ = (c/β) (r̃ − ρ)
- ż = − z [ ρ + A(1−θ) k^{θ−1} − γ z − c/k ]

Capital tax rate along the path: τₖ = 1 − r̃/(r − δ) (guarded if r ≈ δ). 

## Steady state (positive-interest regime r̃* = ρ)

- k* = (A θ / (ρ + δ))^{1/(1−θ)}
- c* = ρ k* + A η (k*)^θ
- x* = A(1−η) (k*)^θ − (δ + ρ) k*
- λ* = γ / (k* [A(1−η) (k*)^{θ−1} − δ − ρ])
- μ* = ( (c*)^{−β} − λ* ) / ρ
- τₖ* = 1 − r̃*/(r − δ) with r̃* = ρ

Interior feasibility at k*: A(1−η) (k*)^{θ−1} − δ − ρ > 0 and x* > 0.

## Solver and diagnostics

- Method: continuation + BVP collocation (MIRK6) satisfying k(0)=k0, k(T)=k*, and ċ(T)=0 (⇒ r̃(T)=ρ); IVP shooting fallback.
- Guards: positivity for k, c, z; derivative clamping; safe powers; small initial dt.
- Transversality diagnostics: e^{−ρt} λ k, e^{−ρt} μ c, and e^{−ρt} c^{−β} k → 0. 
- Success thresholds (terminal): |r̃−ρ|, |k̇|, |ċ|, |k−k*|, |c−c*| small; TVCs < 1e−2.

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

Details: computes W = ∫₀ᵀ [γ log x(t) + c(t)^{1−β}/(1−β)] e^{−ρ t} dt with x = γ/λ and trapezoidal quadrature. Use `limit=` to do a subset; results saved to `welfare_gamma_scan.csv`.

Plot welfare vs γ from the saved CSV (square figure, save-only):

```julia
julia --project=. -e 'include("visualization.jl"); plot_welfare_vs_gamma()'
```

Output: `welfare_vs_gamma.png`.

Stability (eigenvalues of linearization):

```julia
julia --project=. -e 'include("main.jl"); steady_linearization_eigs_kclm()'    # (k,c,λ,μ)
```

Note: μ is recovered from the intratemporal FOC μ = (c^{−β} − λ)/ρ along interior paths; we keep (k,c,z) as states for numerical robustness.

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
