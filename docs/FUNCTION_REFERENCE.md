# Function Reference

This document collects the code-level documentation for the repository in one place.

Conventions used below:

- Scalar means a single number, typically `Float64`.
- A path of length `N` means a vector with one entry per time node.
- In the `OptimalWealthTax` collocation code, a flat collocation vector has length `6N` and stores `(k, c, q, Λ1, Λ2, Λ3)` at each node.
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

### `solveOptimalWealthTaxation(; kwargs...)`

Purpose:
- Public wrapper for the `OptimalWealthTax` runner.

Inputs:
- No positional arguments.

Optional parameters:
- `kwargs...`: forwarded to `_solveOptimalWealthTaxation`, such as `output_dir`, `progress`, `model_kwargs`, and `solve_kwargs`.

Output:
- Returns a named tuple with effective parameters, solve options, the solution object, and generated output paths.
- The stored trajectories typically have length `N`.

### `x(; kwargs...)`

Purpose:
- Legacy alias for `solveOptimalWealthTaxation`.

Inputs:
- No positional arguments.

Optional parameters:
- Same as `solveOptimalWealthTaxation`.

Output:
- Same named tuple returned by `solveOptimalWealthTaxation`.

## `OptimalWealthTax`

### Types

#### `ModelParams(; kwargs...)`

Purpose:
- Stores the parameterization of the `OptimalWealthTax` model.

Optional parameters:
- `A`, `θ`, `η`, `β`, `ρ`, `δ`, `γ`, `n`, `l`: scalar model coefficients.
- `k0`, `q0`, `Λ20`: scalar initial conditions.
- `T`: scalar horizon.
- `N`: number of collocation nodes.
- `max_iter`: nonlinear solver iteration limit.
- `residual_tolerance`: scalar tolerance on the maximum residual.
- `mesh_power`: scalar exponent controlling front-loading of the time grid.
- `min_positive`: scalar numerical floor.

Output:
- Immutable struct containing only scalars.

#### `SteadyStateResult`

Purpose:
- Stores the steady state of the `OptimalWealthTax` model.

Fields:
- `k`, `c`, `q`, `Λ1`, `Λ2`, `Λ3`, `r_tilde`, `x`: all scalars.

Output dimensions:
- Each field has size `1 x 1`.

#### `CollocationResult`

Purpose:
- Stores the output of a collocation solve.

Fields:
- `success`: scalar Boolean.
- `t`, `k`, `c`, `q`, `Λ1`, `Λ2`, `Λ3`, `r_tilde`, `x`: vectors of common length `N`.
- `steady`: one `SteadyStateResult`.
- `residual_norm`: scalar.

### Model and Steady State

#### `production_scale(p)`

Purpose:
- Computes the scale factor of the Cobb-Douglas technology.

Inputs:
- `p::ModelParams`.

Output:
- Scalar `Float64`.

#### `production_terms(k, p)`

Purpose:
- Evaluates production and derivatives at a given capital level.

Inputs:
- `k::Real`: scalar capital.
- `p::ModelParams`.

Output:
- Named tuple with scalar fields `F`, `Fk`, `Fn`, `Fkk`, `Fnk`.

#### `smooth_positive_part(z, ε)`

Purpose:
- Smooth approximation of `max(0, z)`.

Inputs:
- `z::Real`: scalar.
- `ε::Real`: scalar smoothing parameter.

Output:
- One scalar.

#### `foc_implied_controls(y, p)`

Purpose:
- Reconstructs implied controls from the first-order conditions.

Inputs:
- `y::AbstractVector{<:Real}`: uses at least the first five entries `(k, c, q, Λ1, Λ2)`.
- `p::ModelParams`.

Output:
- Either `nothing` or a named tuple with scalar fields `r_tilde`, `x`, `F`, `Fk`, `Fn`, `Fkk`, `Fnk`.

#### `dynamics(y, p)`

Purpose:
- Evaluates the six-dimensional ODE system.

Inputs:
- `y`: vector of length 6 ordered as `(k, c, q, Λ1, Λ2, Λ3)`.
- `p::ModelParams`.

Output:
- `Vector{Float64}` of length 6.

#### `find_steady_state()`
#### `find_steady_state(p)`

Purpose:
- Computes the steady state with default or user-supplied parameters.

Inputs:
- None for the zero-argument version.
- `p::ModelParams` for the parameterized version.

Output:
- `SteadyStateResult` containing scalar fields.

### Low-Level Collocation Helpers

#### `print_progress_result(label, result)`

Purpose:
- Prints a short summary of a collocation result.

Inputs:
- `label::AbstractString`.
- `result::CollocationResult`.

Output:
- Returns `nothing`.

#### `transversality_metrics(result, p)`

Purpose:
- Computes discounted transversality diagnostics along a collocation path.

Inputs:
- `result::CollocationResult` with path length `N`.
- `p::ModelParams`.

Output:
- Named tuple with vector fields `k`, `c`, `q`, each of length `N`, plus scalar terminal values in `terminal`.

#### `pack_solution(result)`

Purpose:
- Packs a structured solution into the flat vector used by the nonlinear system.

Inputs:
- `result::CollocationResult` with path length `N`.

Output:
- `Vector{Float64}` of length `6N`.

#### `with_initial_conditions(p, k0, q0; Λ20=p.Λ20)`

Purpose:
- Copies a parameter object while replacing initial conditions.

Inputs:
- `p::ModelParams`.
- `k0::Real`, `q0::Real`: scalars.

Optional parameters:
- `Λ20::Real`: scalar initial `Λ2`.

Output:
- `ModelParams` object.

#### `with_horizon(p, T, N)`

Purpose:
- Copies a parameter object while replacing horizon and mesh size.

Inputs:
- `p::ModelParams`.
- `T::Real`: scalar horizon.
- `N::Integer`: scalar node count.

Output:
- `ModelParams` object.

#### `node_offset(i)`

Purpose:
- Returns the offset of node `i` in a flat collocation vector.

Inputs:
- `i::Int`.

Output:
- Integer scalar.

#### `node_slice(z, i)`

Purpose:
- Returns the six variables stored at node `i`.

Inputs:
- `z`: flat vector of length `6N`.
- `i::Int`.

Output:
- Vector view of length 6.

#### `collocation_guess_values(k, q, Λ2, steady, p)`

Purpose:
- Builds local guesses for `c`, `Λ1`, and `Λ3` at one node.

Inputs:
- `k`, `q`, `Λ2`: scalars.
- `steady::SteadyStateResult`.
- `p::ModelParams`.

Output:
- Three scalars `(c_guess, Λ1_guess, Λ3_guess)`.

#### `collocation_guess(p, steady, tgrid)`

Purpose:
- Builds the initial flat collocation guess on a grid.

Inputs:
- `p::ModelParams`.
- `steady::SteadyStateResult`.
- `tgrid`: vector of length `N`.

Output:
- Flat vector of length `6N`.

#### `interpolate_guess(old_t, old_z, new_t)`

Purpose:
- Interpolates a flat collocation vector onto a new grid.

Inputs:
- `old_t`: vector of length `N_old`.
- `old_z`: vector of length `6N_old`.
- `new_t`: vector of length `N_new`.

Output:
- Flat vector of length `6N_new`.

#### `interpolate_state(old_t, old_z, t)`

Purpose:
- Interpolates a single six-dimensional node state.

Inputs:
- `old_t`: vector of length `N`.
- `old_z`: vector of length `6N`.
- `t::Real`.

Output:
- Vector of length 6.

#### `rescale_time_grid(old_t, new_T)`

Purpose:
- Rescales an existing grid to end at `new_T`.

Inputs:
- `old_t`: vector of length `N`.
- `new_T::Real`.

Output:
- Vector of length `N`.

#### `collocation_grid(p, N)`

Purpose:
- Builds the front-loaded collocation time grid.

Inputs:
- `p::ModelParams`.
- `N::Int`.

Output:
- Vector of length `N`.

#### `terminal_residual_values(yT, p, steady, scales, terminal_time, terminal_mode)`

Purpose:
- Evaluates the terminal residual block for the chosen closure rule.

Inputs:
- `yT`: vector of length 6.
- `p::ModelParams`.
- `steady::SteadyStateResult`.
- `scales`: tuple of six scalar normalization factors.
- `terminal_time::Real`.
- `terminal_mode::Symbol`.

Output:
- Vector of length 3.

#### `terminal_residuals!(residual, idx, yT, p, steady, scales, terminal_time, terminal_mode; terminal_alpha=1.0)`

Purpose:
- Writes terminal residuals in place into a larger residual vector.

Inputs:
- `residual`: vector being modified.
- `idx::Int`: starting index.
- `yT`: vector of length 6.
- `p::ModelParams`.
- `steady::SteadyStateResult`.
- `scales`: tuple of six scalars.
- `terminal_time::Real`.
- `terminal_mode::Symbol`.

Optional parameters:
- `terminal_alpha::Float64`: TVC homotopy weight.

Output:
- Returns `nothing`.

#### `collocation_residual!(residual, z, p, steady, tgrid; terminal_mode=:steady_state, terminal_alpha=1.0)`

Purpose:
- Assembles the full collocation residual vector in place.

Inputs:
- `residual`: vector of length `6N`.
- `z`: flat vector of length `6N`.
- `p::ModelParams`.
- `steady::SteadyStateResult`.
- `tgrid`: vector of length `N`.

Optional parameters:
- `terminal_mode::Symbol`.
- `terminal_alpha::Float64`.

Output:
- Returns `nothing`.

#### `unpack_solution(z, p, steady, tgrid, residual_norm, success)`

Purpose:
- Converts a flat vector into a structured `CollocationResult`.

Inputs:
- `z`: vector of length `6N`.
- `p::ModelParams`.
- `steady::SteadyStateResult`.
- `tgrid`: vector of length `N`.
- `residual_norm::Real`: scalar.
- `success::Bool`.

Output:
- `CollocationResult` with path length `N`.

#### `evaluate_candidate(z, p, steady, tgrid; success=false, terminal_mode=:steady_state, terminal_alpha=1.0)`

Purpose:
- Recomputes the residual of a candidate and wraps it as a structured result.

Inputs:
- `z`: vector of length `6N`.
- `p::ModelParams`.
- `steady::SteadyStateResult`.
- `tgrid`: vector of length `N`.

Optional parameters:
- `success::Bool`.
- `terminal_mode::Symbol`.
- `terminal_alpha::Float64`.

Output:
- `CollocationResult` with path length `N`.

#### `solve_nonlinear_system(residual!, guess, p)`

Purpose:
- Runs `NLsolve.nlsolve` for the collocation system.

Inputs:
- `residual!`: in-place callback.
- `guess`: vector, typically length `6N`.
- `p::ModelParams`.

Output:
- `NLsolve` result object or `nothing` if the solve throws.

### High-Level `OptimalWealthTax` Solvers

#### `solve_collocation_problem(p, steady; N=p.N, progress=true, initial_t=nothing, initial_z=nothing, use_mesh_continuation=true, terminal_mode=:steady_state, terminal_alpha=1.0)`

Purpose:
- Solves one collocation problem, optionally using a coarse-to-fine mesh sequence.

Inputs:
- `p::ModelParams`.
- `steady::SteadyStateResult`.

Optional parameters:
- `N::Int`.
- `progress::Bool`.
- `initial_t`: optional previous grid of length `N_prev`.
- `initial_z`: optional previous flat vector of length `6N_prev`.
- `use_mesh_continuation::Bool`.
- `terminal_mode::Symbol`.
- `terminal_alpha::Float64`.

Output:
- Tuple `(final_result, previous_t, previous_z)`.
- `previous_t` has length `N_last`, `previous_z` has length `6N_last`.

#### `continue_horizon(p, steady; N=p.N, progress=true, target_T=p.T, terminal_mode=:steady_state, terminal_alpha=1.0)`

Purpose:
- Performs continuation in the time horizon.

Inputs:
- `p::ModelParams`.
- `steady::SteadyStateResult`.

Optional parameters:
- `N::Int`.
- `progress::Bool`.
- `target_T::Real`.
- `terminal_mode::Symbol`.
- `terminal_alpha::Float64`.

Output:
- Tuple `(final_result, previous_t, previous_z)`.

#### `continue_horizon_stages(p, steady, T_stages; N=p.N, progress=true, terminal_mode=:steady_state, terminal_alpha=1.0)`

Purpose:
- Performs continuation over an explicit list of horizon stages.

Inputs:
- `p::ModelParams`.
- `steady::SteadyStateResult`.
- `T_stages`: vector of length `S` containing scalar horizons.

Optional parameters:
- `N::Int`.
- `progress::Bool`.
- `terminal_mode::Symbol`.
- `terminal_alpha::Float64`.

Output:
- Tuple `(final_result, previous_t, previous_z)`.

#### `solve_collocation_staged_horizon(p, T_stages; N=p.N, progress=true, terminal_mode=:state_steady_state)`

Purpose:
- Convenience wrapper around explicit staged horizon continuation.

Inputs:
- `p::ModelParams`.
- `T_stages`: vector of stage horizons.

Optional parameters:
- `N::Int`.
- `progress::Bool`.
- `terminal_mode::Symbol`.

Output:
- Tuple `(final_result, previous_t, previous_z)`.

#### `solve_bvp_problem(p, steady; N=p.N, progress=true, initial_t, initial_z, terminal_mode=:steady_state)`

Purpose:
- Runs a BVP refinement solve and projects the result back to the collocation grid.

Inputs:
- `p::ModelParams`.
- `steady::SteadyStateResult`.

Optional parameters:
- `N::Int`.
- `progress::Bool`.
- `initial_t`: seed grid of length `N_init`.
- `initial_z`: flat seed vector of length `6N_init`.
- `terminal_mode::Symbol`.

Output:
- `CollocationResult` with path length `N`.

#### `compare_solution_paths(reference, candidate)`

Purpose:
- Compares two collocation solutions on a common grid.

Inputs:
- `reference::CollocationResult`.
- `candidate::CollocationResult`.

Output:
- Named tuple containing scalar residuals, scalar success flags, terminal changes, and normalized path deviations.

#### `refine_with_bvp(reference, p; N=length(reference.t), progress=true, terminal_mode=:state_steady_state)`

Purpose:
- Refines an existing collocation path with the BVP solver and returns verification diagnostics.

Inputs:
- `reference::CollocationResult`.
- `p::ModelParams`.

Optional parameters:
- `N::Int`.
- `progress::Bool`.
- `terminal_mode::Symbol`.

Output:
- Named tuple with fields `reference`, `refined`, `verification`, and `preserved_reference`.

#### `continue_initial_conditions(p, steady, previous_t, previous_z; N=p.N, progress=true, base_step=0.025, min_step=1e-4, terminal_mode=:steady_state, terminal_alpha=1.0, label, endpoint)`

Purpose:
- Performs homotopy continuation in initial conditions.

Inputs:
- `p::ModelParams`.
- `steady::SteadyStateResult`.
- `previous_t`: grid of length `N_prev`.
- `previous_z`: flat vector of length `6N_prev`.

Optional parameters:
- `N::Int`.
- `progress::Bool`.
- `base_step::Float64`.
- `min_step::Float64`.
- `terminal_mode::Symbol`.
- `terminal_alpha::Float64`.
- `label::AbstractString`.
- `endpoint`: callable mapping `α` to `(k0, q0)` or `(k0, q0, Λ20)`.

Output:
- Tuple `(current_alpha, final_result, previous_t, previous_z)`.
- `current_alpha` is scalar, `final_result` is a `CollocationResult`, `previous_z` has length `6N_last`.

#### `continue_terminal_conditions(p, steady, previous_t, previous_z; N=p.N, progress=true, base_step=0.1, min_step=1e-5, acceptance_tolerance=...)`

Purpose:
- Performs homotopy from state-based terminal conditions to TVC conditions.

Inputs:
- `p::ModelParams`.
- `steady::SteadyStateResult`.
- `previous_t`: grid of length `N_prev`.
- `previous_z`: flat vector of length `6N_prev`.

Optional parameters:
- `N::Int`.
- `progress::Bool`.
- `base_step::Float64`.
- `min_step::Float64`.
- `acceptance_tolerance`: scalar acceptance threshold.

Output:
- Tuple `(current_alpha, final_result, previous_t, previous_z)`.

#### `better_target_result(lhs, rhs)`

Purpose:
- Chooses the better result between two collocation candidates.

Inputs:
- `lhs::CollocationResult`.
- `rhs::CollocationResult`.

Output:
- One `CollocationResult`.

#### `solve_collocation(p=ModelParams(); N=p.N, progress=true, use_continuation=true, use_bvp_refinement=false, use_horizon_continuation=false, terminal_mode=:state_steady_state, use_nested_seed=true)`

Purpose:
- Main high-level entry point for the `OptimalWealthTax` collocation solver.

Inputs:
- `p::ModelParams = ModelParams()`.

Optional parameters:
- `N::Int`.
- `progress::Bool`.
- `use_continuation::Bool`.
- `use_bvp_refinement::Bool`.
- `use_horizon_continuation::Bool`.
- `terminal_mode::Symbol`.
- `use_nested_seed::Bool`.

Output:
- One `CollocationResult`.
- All trajectory fields have the common length of the final grid used by the solve attempt.

### Plotting and Runner

#### `plot_solution(result, title, filename; force=false, half=false)`

Purpose:
- Saves a nine-panel plot of the collocation solution.

Inputs:
- `result::CollocationResult` with path length `N`.
- `title::AbstractString`.
- `filename::AbstractString`.

Optional parameters:
- `force::Bool`.
- `half::Bool`.

Output:
- Returns `filename` on success, `nothing` if plotting is skipped.

#### `default_optimal_wealth_taxation_output_dir()`

Purpose:
- Returns the default output directory string for `OptimalWealthTax` runs.

Output:
- One path string.

#### `writeOptimalWealthTaxationResultCSV(result, p, file_path)`

Purpose:
- Writes the full collocation path and TVC diagnostics to CSV.

Inputs:
- `result::CollocationResult` with path length `N`.
- `p::ModelParams`.
- `file_path::AbstractString`.

Output:
- Returns `file_path`.
- Writes `N` rows and 12 columns.

#### `_solveOptimalWealthTaxation(; output_dir=..., progress=true, model_kwargs=(;), solve_kwargs=(;))`

Purpose:
- Full runner: solve, save CSV, save summary CSV, and save plot.

Inputs:
- No positional arguments.

Optional parameters:
- `output_dir::AbstractString`.
- `progress::Bool`.
- `model_kwargs::NamedTuple`.
- `solve_kwargs::NamedTuple`.

Output:
- Named tuple with fields `params`, `solve_kwargs`, `result`, `solution_csv`, `plot_png`, and `summary_csv`.

## `NoWealthTaxation`

### Types and Steady State

#### `ModelParams(; kwargs...)`

Purpose:
- Parameter container for the legacy `NoWealthTaxation` model.

Optional parameters:
- `A`, `θ`, `η`, `ρ`, `β`, `δ`, `γ`, `r`, `k0`, `T`: all scalars.

Output:
- Immutable scalar-only struct.

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

#### `solve_orct(p; T=p.T, N=2001, debug=false, progress=true)`

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