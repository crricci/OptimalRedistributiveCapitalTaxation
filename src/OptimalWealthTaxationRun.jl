default_optimal_wealth_taxation_output_dir() = joinpath(normpath(joinpath(@__DIR__, "..")), "outputs", "optimal_wealth_taxation")

function writeOptimalWealthTaxationResultCSV(result::OptimalWealthTax.CollocationResult, file_path::AbstractString)
    headers = ["t", "k", "c", "q", "Lambda1", "Lambda2", "Lambda3", "r_tilde", "x"]
    rows = ((result.t[i], result.k[i], result.c[i], result.q[i], result.Λ1[i], result.Λ2[i], result.Λ3[i], result.r_tilde[i], result.x[i]) for i in eachindex(result.t))
    return write_csv_table(file_path, headers, rows)
end

function _solveOptimalWealthTaxation(; output_dir::AbstractString = default_optimal_wealth_taxation_output_dir(), progress::Bool = true, model_kwargs::NamedTuple = (;), solve_kwargs::NamedTuple = (;))
    mkpath(output_dir)
    p = OptimalWealthTax.ModelParams(; model_kwargs...)
    result = OptimalWealthTax.solve_collocation(p; progress = progress, solve_kwargs...)

    csv_path = joinpath(output_dir, "OptimalWealthTaxation_solution.csv")
    png_path = joinpath(output_dir, "OptimalWealthTaxation_solution.png")
    summary_path = joinpath(output_dir, "OptimalWealthTaxation_summary.csv")

    writeOptimalWealthTaxationResultCSV(result, csv_path)
    OptimalWealthTax.plot_solution(result, "OptimalWealthTaxation path", png_path; force = true)
    write_csv_table(summary_path, ["success", "residual_norm", "final_k", "final_c", "final_q", "solution_csv", "plot_png"], [(result.success, result.residual_norm, result.k[end], result.c[end], result.q[end], csv_path, png_path)])

    return (; params = p, result = result, solution_csv = csv_path, plot_png = png_path, summary_csv = summary_path)
end