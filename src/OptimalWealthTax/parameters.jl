"""
    ModelParams(; kwargs...)

Parameter container for the `OptimalWealthTax` model.

Input arguments:
- No positional arguments.

Optional parameters:
- `A`, `θ`, `η`, `β`, `ρ`, `δ`, `γ`, `n`, `l`: `Float64` scalars defining technology, preferences, and resource constraints.
- `k0`, `q0`, `Λ20`: scalar initial conditions for the two states and the `Λ2` costate.
- `T`: scalar time horizon.
- `N`: number of collocation grid nodes.
- `max_iter`: maximum number of nonlinear-solver iterations.
- `residual_tolerance`: tolerance on the maximum absolute residual.
- `mesh_power`: exponent controlling the front-loaded time grid.
- `min_positive`: numerical floor used to avoid divisions by zero.

Output:
- Returns an immutable struct containing only scalars.
- It contains no vectors; trajectory lengths are determined later by `N`.
"""
@with_kw struct ModelParams
    A::Float64 = 1.0
    θ::Float64 = 0.3
    η::Float64 = 0.6
    β::Float64 = 0.75
    ρ::Float64 = 0.03
    δ::Float64 = 0.1
    γ::Float64 = 0.5
    n::Float64 = 1.0
    l::Float64 = 1.0
    k0::Float64 = 2.0
    q0::Float64 = 1.0
    Λ20::Float64 = 0.0
    T::Float64 = 40.0
    N::Int = 21
    max_iter::Int = 800
    residual_tolerance::Float64 = 1e-6
    mesh_power::Float64 = 4.0
    min_positive::Float64 = 1e-10
end