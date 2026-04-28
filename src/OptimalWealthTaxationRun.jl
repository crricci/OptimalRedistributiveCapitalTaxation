"""
    default_optimal_wealth_taxation_output_dir()

Returns the default output directory for `OptimalWealthTax` runs.

Input arguments:
- None.

Optional parameters:
- None.

Output:
- Returns an `AbstractString` path.
- The output is a single path string, not a collection.
"""
default_optimal_wealth_taxation_output_dir() = joinpath(normpath(joinpath(@__DIR__, "..")), "outputs", "optimal_wealth_taxation")

"""
    writeOptimalWealthTaxationResultCSV(result, p, file_path)

Writes the full `OptimalWealthTax` trajectory and transversality diagnostics to CSV.

Input arguments:
- `result::OptimalWealthTax.CollocationResult`: solution object with trajectory length `N = length(result.t)`.
- `p::OptimalWealthTax.ModelParams`: model parameters.
- `file_path::AbstractString`: output CSV path.

Optional parameters:
- None.

Output:
- Returns `file_path`.
- Writes `N` data rows and 12 columns: time, six primal/costate variables, two controls, and three TVC diagnostics.
"""
function writeOptimalWealthTaxationResultCSV(result::OptimalWealthTax.CollocationResult, p::OptimalWealthTax.ModelParams, file_path::AbstractString)
    tvc = OptimalWealthTax.transversality_metrics(result, p)
    headers = ["t", "k", "c", "q", "Lambda1", "Lambda2", "Lambda3", "r_tilde", "x", "tvc_k", "tvc_c", "tvc_q"]
    rows = ((result.t[i], result.k[i], result.c[i], result.q[i], result.Λ1[i], result.Λ2[i], result.Λ3[i], result.r_tilde[i], result.x[i], tvc.k[i], tvc.c[i], tvc.q[i]) for i in eachindex(result.t))
    return write_csv_table(file_path, headers, rows)
end

"""
    _solveOptimalWealthTaxation(; output_dir=default_optimal_wealth_taxation_output_dir(), progress=true, model_kwargs=(;), solve_kwargs=(;))

Runs the full `OptimalWealthTax` workflow: solve, save CSV, save summary CSV, and save plot.

Input arguments:
- No positional arguments.

Optional parameters:
- `output_dir::AbstractString`: directory where outputs are written.
- `progress::Bool = true`: print run progress.
- `model_kwargs::NamedTuple = (; )`: keyword overrides passed to `OptimalWealthTax.ModelParams`.
- `solve_kwargs::NamedTuple = (; )`: keyword overrides passed to `OptimalWealthTax.solve_collocation`.

Output:
- Returns a named tuple with fields `params`, `solve_kwargs`, `result`, `solution_csv`, `plot_png`, and `summary_csv`.
- `result` is a `CollocationResult` whose trajectory fields have length `N = params.N` unless the solver exits on a different final stage.
"""
function _solveOptimalWealthTaxation(; output_dir::AbstractString = default_optimal_wealth_taxation_output_dir(), progress::Bool = true, model_kwargs::NamedTuple = (;), solve_kwargs::NamedTuple = (;))
    mkpath(output_dir)
    default_model_kwargs = (; T = 120.0, N = 81, max_iter = 1800)
    default_solve_kwargs = (; terminal_mode = :costate_steady_state, use_nested_seed = false)
    effective_model_kwargs = merge(default_model_kwargs, model_kwargs)
    effective_solve_kwargs = merge(default_solve_kwargs, solve_kwargs)

    progress && println("OptimalWealthTaxation run start")
    progress && println("  output_dir=$(output_dir)")
    progress && println("  model_kwargs=$(effective_model_kwargs)")
    progress && println("  solve_kwargs=$(effective_solve_kwargs)")

    p = OptimalWealthTax.ModelParams(; effective_model_kwargs...)
    result = OptimalWealthTax.solve_collocation(p; progress = progress, effective_solve_kwargs...)
    tvc = OptimalWealthTax.transversality_metrics(result, p)

    csv_path = joinpath(output_dir, "OptimalWealthTaxation_solution.csv")
    png_path = joinpath(output_dir, "OptimalWealthTaxation_solution.png")
    summary_path = joinpath(output_dir, "OptimalWealthTaxation_summary.csv")

    writeOptimalWealthTaxationResultCSV(result, p, csv_path)
    OptimalWealthTax.plot_solution(result, "OptimalWealthTaxation path", png_path; force = true)
    write_csv_table(summary_path,
        ["success", "residual_norm", "terminal_mode", "horizon_T", "mesh_N", "final_k", "steady_k", "final_c", "steady_c", "final_q", "steady_q", "terminal_tvc_k", "terminal_tvc_c", "terminal_tvc_q", "solution_csv", "plot_png"],
        [(result.success, result.residual_norm, String(effective_solve_kwargs.terminal_mode), p.T, p.N, result.k[end], result.steady.k, result.c[end], result.steady.c, result.q[end], result.steady.q, tvc.terminal.k, tvc.terminal.c, tvc.terminal.q, csv_path, png_path)])

    if progress
        println("OptimalWealthTaxation run complete")
        println("  success=$(result.success) residual=$(result.residual_norm)")
        println("  final  k=$(result.k[end]) c=$(result.c[end]) q=$(result.q[end])")
        println("  steady k=$(result.steady.k) c=$(result.steady.c) q=$(result.steady.q)")
        println("  tvc    k=$(tvc.terminal.k) c=$(tvc.terminal.c) q=$(tvc.terminal.q)")
        println("  solution_csv=$(csv_path)")
        println("  summary_csv=$(summary_path)")
        println("  plot_png=$(png_path)")
    end

    return (; params = p, solve_kwargs = effective_solve_kwargs, result = result, solution_csv = csv_path, plot_png = png_path, summary_csv = summary_path)
end