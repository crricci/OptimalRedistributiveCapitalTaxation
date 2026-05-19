# Function Reference

This document collects the code-level documentation for the repository in one place.

Conventions used below:

- Scalar means a single number, typically `Float64`.
- A path of length `N` means a vector with one entry per time node.
- In the legacy `NoWealthTaxation` solver, trajectory vectors all share the same length `N = length(t)`.

## Entry Points

### `solveNoWealthTaxation(; kwargs...)`

Purpose:
- Public wrapper for the `NoWealthTaxation` runner.

Inputs:
- No positional arguments.

Optional parameters:
- `kwargs...`: forwarded to `_solveNoWealthTaxation`, including `k0_values`, `output_dir`, `model_kwargs`, `solve_kwargs`, and scan options.

Output:
- Returns a named tuple containing runner results, saved file paths, and optional scan outputs.
- Any numerical trajectories inside the result have length determined by the underlying solver.

### `solveReducedOptimalWealthTaxation(; kwargs...)`

Purpose:
- Public wrapper for the reduced `OptimalWealthTaxation` runner.

Inputs:
- No positional arguments.

Optional parameters:
- `kwargs...`: forwarded to `_solveReducedOptimalWealthTaxation`, such as `output_dir`, `progress`, `model_kwargs`, and continuation settings.

Output:
- Returns a named tuple with reduced-model parameters, the optimized path, and generated CSV summaries.
- The stored trajectories have length `N`.

## `OptimalWealthTaxationReduced`

### Types and Core Helpers

#### `ReducedOptimalWealthTaxParams(; kwargs...)`

Purpose:
- Compatibility alias for the shared `ModelParams` container.

Optional parameters:
- See `ModelParams`; the reduced solver now uses the same shared parameter container as `NoWealthTaxation`.

Output:
- Same struct as `ModelParams`.

#### `reduced_time_grid(p)`

Purpose:
- Builds the reduced solver time grid.

Inputs:
- `p::ReducedOptimalWealthTaxParams`.

Output:
- Vector of length `N`.

#### `reduced_production_terms(k, p)`

Purpose:
- Evaluates reduced-form production objects at a scalar capital level.

Inputs:
- `k::Real`.
- `p::ReducedOptimalWealthTaxParams`.

Output:
- Named tuple with scalar fields `F`, `Fk`, `Fn`.

#### `reduced_interpolate_control_guess(old_t, old_r, new_t)`

Purpose:
- Interpolates a control seed from one reduced grid to another.

Inputs:
- `old_t`, `old_r`, `new_t`: vectors.

Output:
- Vector with length `length(new_t)`.

#### `solve_reduced_optimal_wealth_taxation_stage(p; initial_r=nothing)`

Purpose:
- Solves one reduced NLP stage.

Inputs:
- `p::ReducedOptimalWealthTaxParams`.

Output:
- Named tuple with reduced-solver trajectories and diagnostics.

#### `_solveReducedOptimalWealthTaxation(; output_dir=..., progress=true, model_kwargs=(;), use_continuation=nothing, continuation_T_stages=nothing, continuation_N_stages=nothing, continuation_max_T_step=nothing, continuation_min_T_step=nothing)`

Purpose:
- Full reduced runner: solve, save trajectory CSV, save summary CSV, and save continuation-stage CSV.

Inputs:
- No positional arguments.

Optional parameters:
- `output_dir::AbstractString`.
- `progress::Bool`.
- `model_kwargs::NamedTuple`.
- `use_continuation`: optional Bool override.
- `continuation_T_stages`, `continuation_N_stages`: optional explicit stage lists.
- `continuation_max_T_step`, `continuation_min_T_step`: optional adaptive step controls.

Output:
- Named tuple with fields `params`, `result`, `solution_csv`, `summary_csv`, `stages_csv`, and `stage_rows`.

## `NoWealthTaxation`

### Types and Steady State

#### `ModelParams(; kwargs...)`

Purpose:
- Parameter container for the legacy `NoWealthTaxation` model.

Optional parameters:
- Shared economic section: `A`, `θ`, `η`, `ρ`, `β`, `δ`, `γ`.
- Additional economic section for `OptimalWealthTaxation`: `n`, `l`.
- Shared state and horizon section: `k0`, `N`, `T`.
- Additional state section for `OptimalWealthTaxation`: `q0`.
- Numerical safeguard used by the active model families: `min_positive`.
- Numerical parameters for `NoWealthTaxation` only: `derivative_clamp`, `solver_failure_penalty`.
- Legacy IVP section: `ivp_abstol`, `ivp_reltol`, `ivp_dt_initial_cap`, `ivp_dt_initial_divisor`, `ivp_dtmin`, `ivp_dtmax_floor`, `ivp_dtmax_divisor`, `ivp_maxiters`.
- Legacy shooting section: `shooting_stage_cutoff`, `shooting_stage_schedule_short`, `shooting_stage_schedule_long`, `shooting_seed_multipliers`, `shooting_initial_c_scale`, `shooting_log_floor`, `shooting_c_cap_scale`, `shooting_lambda_cap`, `shooting_xtol`, `shooting_ftol`, `shooting_iterations`, `shooting_guess_blend_old_weight`, `shooting_guess_blend_new_weight`.
- Legacy BVP section: `bvp_abstol`, `bvp_reltol`, `bvp_dt_floor`, `bvp_dt_divisor`.
- Legacy terminal diagnostics section: `terminal_r_tolerance`, `terminal_kdot_tolerance`, `terminal_cdot_tolerance`, `terminal_k_tolerance`, `terminal_c_tolerance`, `transversality_tolerance`.
- Numerical parameters for `OptimalWealthTaxation` only: `max_iter`, `ipopt_print_level`, `use_continuation`, `continuation_initial_T`, `continuation_initial_N`, `continuation_max_T_step`, `continuation_min_T_step`.

Output:
- Immutable struct containing scalar parameters plus fixed-size tuple schedules.

#### `SteadyStateResult`

Purpose:
- Stores the analytical steady state of the legacy model.

Fields:
- `k`, `c`, `λ`, `μ`, `r_tilde`, `x`, `tau_k`: all scalars.

#### `analytic_steady_state(p)`

Purpose:
- Computes the analytical steady state.

Inputs:
- `p`, usually `ModelParams`.

Output:
- `SteadyStateResult` with scalar fields.

#### `find_steady_state`

Purpose:
- Alias to `analytic_steady_state`.

Output:
- `SteadyStateResult`.

### Legacy Solver and Diagnostics

#### `SolutionResult`

Purpose:
- Stores the legacy solver output.

Fields:
- `success`: scalar Boolean.
- `t`, `k`, `c`, `λ`, `μ`, `r_tilde`, `tau_k`, `λ_tr`, `μ_tr`, `c_tr`: vectors of common length `N`.
- `steady`: scalar steady-state object.

#### `check_residuals(res; p=ModelParams())`

Purpose:
- Evaluates residual diagnostics on a discrete legacy solution path.

Inputs:
- `res::SolutionResult` with path length `N`.

Optional parameters:
- `p`: parameter object.

Output:
- Dictionary of scalar max and RMS norms.

#### `solve_orct(p; T=p.T, N=p.N, debug=false, progress=true)`

Purpose:
- Main legacy solver based on shooting continuation plus a BVP attempt.

Inputs:
- `p`, usually `ModelParams`.

Optional parameters:
- `T`: scalar horizon.
- `N::Int`: number of stored time points for IVP output.
- `debug::Bool`.
- `progress::Bool`.

Output:
- `SolutionResult`.
- All stored path vectors have common length `N_path`.

### Legacy Plotting

#### `plot_main_solution(result, title, filename; force=false, half=true)`

Purpose:
- Saves a nine-panel plot of the legacy solution.

Inputs:
- `result`: solution object with path length `N`.
- `title`.
- `filename`.

Optional parameters:
- `force::Bool`.
- `half::Bool`.

Output:
- Returns `nothing`.

#### `plot_welfare_vs_gamma(csvfile="welfare_gamma_scan.csv", outfile="welfare_vs_gamma.png"; title="Welfare vs γ")`

Purpose:
- Plots welfare against `γ` from a two-column CSV.

Inputs:
- `csvfile`: CSV path with `M` data rows.
- `outfile`: PNG path.

Optional parameters:
- `title`.

Output:
- Returns `outfile` if at least two valid rows exist, otherwise `nothing`.

#### `plot_gamma_vs_kstar(csvfile="gamma_kstar_scan.csv", outfile="gamma_vs_kstar.png"; title="Steady State Capital k* vs γ")`

Purpose:
- Plots steady-state capital against `γ` from a two-column CSV.

Inputs:
- `csvfile`: CSV path with `M` data rows.
- `outfile`: PNG path.

Optional parameters:
- `title`.

Output:
- Returns `outfile` if at least two valid rows exist, otherwise `nothing`.

#### `plot_gamma_vs_steadystate_welfare(csvfile="gamma_steadystate_welfare_scan.csv", outfile="gamma_vs_steadystate_welfare.png"; title="Steady State Welfare vs γ")`

Purpose:
- Plots steady-state welfare against `γ` from a two-column CSV.

Inputs:
- `csvfile`: CSV path with `M` data rows.
- `outfile`: PNG path.

Optional parameters:
- `title`.

Output:
- Returns `outfile` if at least two valid rows exist, otherwise `nothing`.

### Legacy Analysis Helpers

#### `compute_residuals(p, sol)`

Purpose:
- Computes pointwise residual vectors for the legacy model equations.

Inputs:
- `p::NoWealthTaxation.ModelParams`.
- `sol::NoWealthTaxation.SolutionResult` with path length `N`.

Output:
- Named tuple whose fields are vectors of length `N`.

#### `summarize_residuals(label, R)`

Purpose:
- Prints max and mean absolute residual summaries.

Inputs:
- `label`.
- `R`: residual bundle with vectors of common length `N`.

Output:
- Returns `nothing`.

#### `test_steady_state_invariance(p; T=50.0)`

Purpose:
- Solves the legacy model from the analytical steady state and checks invariance.

Inputs:
- `p::NoWealthTaxation.ModelParams`.

Optional parameters:
- `T = 50.0`.

Output:
- `SolutionResult` on success, otherwise `nothing`.

#### `test_perturbations(p; factors=[0.9, 1.1])`

Purpose:
- Solves the legacy model from perturbations around steady-state capital.

Inputs:
- `p::NoWealthTaxation.ModelParams`.

Optional parameters:
- `factors`: vector of perturbation multipliers.

Output:
- Returns `nothing`.

#### `run_gamma_welfare_scan(; k0=2.0, gamma_values=[...], limit=0, outfile="welfare_gamma_scan.csv", progress=true)`

Purpose:
- Runs the legacy transition solver over a grid of `γ` values and computes discounted welfare.

Optional parameters:
- `k0::Float64`.
- `gamma_values::Vector{Float64}` of length `G`.
- `limit::Int`.
- `outfile::AbstractString`.
- `progress::Bool`.

Output:
- Returns `outfile`.
- Writes a two-column CSV with `G_used` rows.

#### `run_gamma_kstar_scan(; gamma_values=[...], outfile="gamma_kstar_scan.csv", progress=true)`

Purpose:
- Computes steady-state capital over a grid of `γ` values.

Optional parameters:
- `gamma_values::Vector{Float64}` of length `G`.
- `outfile::AbstractString`.
- `progress::Bool`.

Output:
- Returns `outfile`.
- Writes a two-column CSV with `G` rows.

#### `run_gamma_steadystate_welfare_scan(; gamma_values=[...], outfile="gamma_steadystate_welfare_scan.csv", progress=true)`

Purpose:
- Computes steady-state welfare over a grid of `γ` values.

Optional parameters:
- `gamma_values::Vector{Float64}` of length `G`.
- `outfile::AbstractString`.
- `progress::Bool`.

Output:
- Returns `outfile`.
- Writes a two-column CSV with `G` rows.

#### `steady_linearization_eigs_kclm(p=NoWealthTaxation.ModelParams())`

Purpose:
- Builds the steady-state Jacobian of the legacy `(k, c, λ, μ)` system and computes eigenvalues.

Inputs:
- `p::NoWealthTaxation.ModelParams`.

Output:
- Named tuple with:
  - `J`: `4 x 4` matrix.
  - `eigenvalues`: vector of length 4.
  - `classification`: string.
  - `counts`: named tuple of scalar counts.

### Legacy Runner Utilities

#### `write_csv_table(file_path, headers, rows)`

Purpose:
- Writes a generic table to CSV.

Inputs:
- `file_path::AbstractString`.
- `headers::Vector{String}` of length `M`.
- `rows`: iterable with one row per record, each expected to have `M` entries.

Output:
- Returns `file_path`.

#### `default_no_wealth_taxation_output_dir()`

Purpose:
- Returns the default output directory for legacy runs.

Output:
- One path string.

#### `writeNoWealthTaxationResultCSV(result, file_path)`

Purpose:
- Writes the full legacy solution path to CSV.

Inputs:
- `result::SolutionResult` with path length `N`.
- `file_path::AbstractString`.

Output:
- Returns `file_path`.
- Writes `N` rows and 10 columns.

#### `_solveNoWealthTaxation(; k0_values=[2.0, 2.2], output_dir=..., progress=true, model_kwargs=(;), solve_kwargs=(;), run_gamma_scans=true, gamma_scan_k0=NaN, gamma_welfare_scan_kwargs=(;), gamma_steadystate_welfare_scan_kwargs=(;))`

Purpose:
- Full legacy runner across multiple initial conditions, with optional `γ` scans.

Optional parameters:
- `k0_values`: vector of length `K`.
- `output_dir::AbstractString`.
- `progress::Bool`.
- `model_kwargs::NamedTuple`.
- `solve_kwargs::NamedTuple`.
- `run_gamma_scans::Bool`.
- `gamma_scan_k0::Real`.
- `gamma_welfare_scan_kwargs::NamedTuple`.
- `gamma_steadystate_welfare_scan_kwargs::NamedTuple`.

Output:
- Named tuple with fields `results`, `summary_csv`, and `gamma_outputs`.
- `results` is a vector of length `K`.

## Notes

- The repository also contains executable scripts in `scripts/`, but they do not define reusable functions.
- Local nested helper functions inside solver bodies are documented indirectly through the enclosing public functions.