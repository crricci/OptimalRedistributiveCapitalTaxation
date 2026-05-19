# Parameters for the active model families

using Parameters

# Model parameters structure using Parameters.jl
"""
    ModelParams(; kwargs...)

Parameter container for the `NoWealthTaxation` model.

This is the single shared parameter container used by both active branches:
- `NoWealthTaxation`
- `OptimalWealthTaxation`

Input arguments:
- No positional arguments.

Optional parameters:
- Shared economic section: `A`, `θ`, `η`, `ρ`, `β`, `δ`, `γ`.
- Additional economic section for `OptimalWealthTaxation`: `n`, `l`.
- Shared state and horizon section: `k0`, `N`, `T`.
- Additional state section for `OptimalWealthTaxation`: `q0`.
- Numerical safeguards used by the active model families: `min_positive`.
- Numerical parameters used only by `NoWealthTaxation`: `derivative_clamp`, `solver_failure_penalty`.
- Legacy IVP section: `ivp_abstol`, `ivp_reltol`, `ivp_dt_initial_cap`, `ivp_dt_initial_divisor`, `ivp_dtmin`, `ivp_dtmax_floor`, `ivp_dtmax_divisor`, `ivp_maxiters`.
- Legacy shooting section: `shooting_stage_cutoff`, `shooting_stage_schedule_short`, `shooting_stage_schedule_long`, `shooting_seed_multipliers`, `shooting_initial_c_scale`, `shooting_log_floor`, `shooting_c_cap_scale`, `shooting_lambda_cap`, `shooting_xtol`, `shooting_ftol`, `shooting_iterations`, `shooting_guess_blend_old_weight`, `shooting_guess_blend_new_weight`.
- Legacy BVP section: `bvp_abstol`, `bvp_reltol`, `bvp_dt_floor`, `bvp_dt_divisor`.
- Legacy terminal diagnostics section: `terminal_r_tolerance`, `terminal_kdot_tolerance`, `terminal_cdot_tolerance`, `terminal_k_tolerance`, `terminal_c_tolerance`, `transversality_tolerance`.
- Numerical parameters used only by `OptimalWealthTaxation`: `max_iter`, `ipopt_print_level`, `use_continuation`, `continuation_initial_T`, `continuation_initial_N`, `continuation_max_T_step`, `continuation_min_T_step`.
- Numerical parameters used only by `OptimalWealthTaxation`: `max_iter`, `ipopt_print_level`, `use_continuation`, `continuation_initial_T`, `continuation_initial_N`, `continuation_max_T_step`, `continuation_min_T_step`, `reduced_implicit_iterations`.

Output:
- Returns an immutable struct containing scalar model parameters plus fixed-size tuple solver schedules.
- It stores no trajectories; time-series dimensions are determined later by the solver.
"""
@with_kw struct ModelParams

    # Shared economic parameters used by both active model families
    A::Float64 = 10.0      # Productivity parameter
    θ::Float64 = 0.3      # Capital elasticity in production
    η::Float64 = 0.6      # Capital share in production
    ρ::Float64 = 0.03     # Discount rate
    β::Float64 = 0.75     # Inverse elasticity of substitution
    δ::Float64 = 0.01      # Depreciation rate
    γ::Float64 = 0.5      # Redistribution cost parameter

    # Additional economic parameters for OptimalWealthTaxation
    n::Float64 = 1.0      # Labor supply / scale term used by the reduced wealth-tax solver
    l::Float64 = 1.0      # Fixed production scale term used by the reduced wealth-tax solver

    # Shared state and horizon for the active model families
    k0::Float64 = 2.0     # Initial capital
    N::Int = 21         # Saved time nodes for the legacy solution path
    T::Float64 = 200.0   # Time horizon for solution

    # Additional state for OptimalWealthTaxation
    q0::Float64 = 100.0     # Initial q state used by the reduced wealth-tax solver

    # Numerical safeguard used by the active model families
    min_positive::Float64 = 1e-12

    ## Numerical parameters for NoWealthTaxation only

    # Generic safeguards shooting/BVP solver
    derivative_clamp::Float64 = 1e6
    solver_failure_penalty::Float64 = 1e6

    # Numerical parameters IVP solve
    ivp_abstol::Float64 = 1e-8
    ivp_reltol::Float64 = 1e-8
    ivp_dt_initial_cap::Float64 = 1e-3
    ivp_dt_initial_divisor::Float64 = 1000.0
    ivp_dtmin::Float64 = 1e-12
    ivp_dtmax_floor::Float64 = 1e-2
    ivp_dtmax_divisor::Float64 = 200.0
    ivp_maxiters::Int = 20_000_000

    # Numerical parameters shooting continuation
    shooting_stage_cutoff::Float64 = 120.0
    shooting_stage_schedule_short::NTuple{4, Float64} = (10.0, 20.0, 40.0, 80.0)
    shooting_stage_schedule_long::NTuple{4, Float64} = (20.0, 40.0, 80.0, 120.0)
    shooting_seed_multipliers::NTuple{3, Float64} = (0.8, 1.0, 1.2)
    shooting_initial_c_scale::Float64 = 0.9
    shooting_log_floor::Float64 = 1e-8
    shooting_c_cap_scale::Float64 = 10.0
    shooting_lambda_cap::Float64 = 10.0
    shooting_xtol::Float64 = 1e-10
    shooting_ftol::Float64 = 1e-10
    shooting_iterations::Int = 800
    shooting_guess_blend_old_weight::Float64 = 0.7
    shooting_guess_blend_new_weight::Float64 = 0.3

    # Numerical parameters BVP refinement
    bvp_abstol::Float64 = 1e-9
    bvp_reltol::Float64 = 1e-9
    bvp_dt_floor::Float64 = 0.02
    bvp_dt_divisor::Float64 = 400.0

    # Numerical tolerances terminal diagnostics
    terminal_r_tolerance::Float64 = 2e-3
    terminal_kdot_tolerance::Float64 = 2e-3
    terminal_cdot_tolerance::Float64 = 2e-3
    terminal_k_tolerance::Float64 = 5e-3
    terminal_c_tolerance::Float64 = 5e-3
    transversality_tolerance::Float64 = 1e-2

    ## Numerical parameters for OptimalWealthTaxation only
    max_iter::Int = 800
    ipopt_print_level::Int = 0
    use_continuation::Bool = true
    continuation_initial_T::Float64 = 1.0
    continuation_initial_N::Int = 21
    continuation_max_T_step::Float64 = 1.0
    continuation_min_T_step::Float64 = 0.25
    reduced_implicit_iterations::Int = 8
end

