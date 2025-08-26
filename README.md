# Optimal Redistributive Capital Taxation (ORCT) - Numerical Solver

This project solves the forward-backward system derived from the Pontryagin Maximum Principle for the ORCT model, computes the steady state, and plots the optimal trajectory.

Contents:
- `parameters.jl` — parameter struct and defaults
- `steady_state.jl` — computes the positive-interest steady state (r̃* = ρ)
- `solver.jl` — BVP collocation solver with r̃ = max{0, ·}
- `visualization.jl` — plotting helper (`plot_main_solution`)
- `main.jl` — runner script

## Steady state (r̃* > 0 regime)

Let r̃* = ρ. Define for k>0:
- c(k) = ρ k + A η k^θ
- x(k) = A(1−η)k^θ − (δ + ρ)k
- λ(k) = γ / (k [A(1−η)k^{θ−1} − δ − ρ])
- μ(k) = (c(k)^{−β} − λ(k)) / ρ

The intratemporal condition implies the scalar equation in k:
  F(k) = λ(k) + μ(k) c(k)/(β k) − γ/x(k) = 0

We solve F(k)=0 for k*, then obtain c*, λ*, μ* from the formulas above.

## How to run

Run the solver and plot:

```bash
julia --project=. main.jl
```

Output: `solution.png` in the project root. The plot shows only [0, T/2] per `visualization.jl`.

## Notes
- The solver is a BVP on [0,T] with `T` taken from `ModelParams` (default 1000). It enforces `k(0)=k0` and terminal conditions approximating transversality and the intratemporal/corner condition at T.
- The kink r̃ = max{0, ·} is handled pointwise. If the solution ends in the corner at T, the boundary condition enforces r̃(T)=0; otherwise it enforces the intratemporal equality.
- Ensure parameters imply a valid positive-interest steady state: A(1−η)k^{θ−1} − δ − ρ > 0 and x(k) > 0 at k=k*.
