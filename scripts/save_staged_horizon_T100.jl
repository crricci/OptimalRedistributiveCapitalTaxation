include(joinpath(@__DIR__, "..", "src", "NoWealthTaxation.jl"))
include(joinpath(@__DIR__, "..", "src", "OptimalWealthTax.jl"))
include(joinpath(@__DIR__, "..", "src", "NoWealthTaxationRun.jl"))
include(joinpath(@__DIR__, "..", "src", "OptimalWealthTaxationRun.jl"))

outdir = joinpath(@__DIR__, "..", "outputs", "optimal_wealth_taxation_T100_staged")
mkpath(outdir)

T_stages = [40.0, 55.0, 70.0, 85.0, 100.0]
p = Main.OptimalWealthTax.ModelParams(T = 100.0, N = 81, max_iter = 1800, mesh_power = 4.0)
result, _, _ = Main.OptimalWealthTax.solve_collocation_staged_horizon(p, T_stages;
    N = p.N,
    progress = true,
    terminal_mode = :state_steady_state)

tvc = Main.OptimalWealthTax.transversality_metrics(result, p)
csv_path = joinpath(outdir, "OptimalWealthTaxation_solution.csv")
png_path = joinpath(outdir, "OptimalWealthTaxation_solution.png")
summary_path = joinpath(outdir, "OptimalWealthTaxation_summary.csv")

writeOptimalWealthTaxationResultCSV(result, p, csv_path)
Main.OptimalWealthTax.plot_solution(result, "OptimalWealthTaxation staged horizon path", png_path; force = true)
write_csv_table(summary_path,
    ["success", "residual_norm", "terminal_mode", "horizon_T", "mesh_N", "T_stages", "final_k", "steady_k", "final_c", "steady_c", "final_q", "steady_q", "terminal_tvc_k", "terminal_tvc_c", "terminal_tvc_q", "solution_csv", "plot_png"],
    [(result.success, result.residual_norm, "state_steady_state", p.T, p.N, join(T_stages, ";"), result.k[end], result.steady.k, result.c[end], result.steady.c, result.q[end], result.steady.q, tvc.terminal.k, tvc.terminal.c, tvc.terminal.q, csv_path, png_path)])

println((; success = result.success, residual = result.residual_norm, T_final = result.t[end], solution_csv = csv_path, plot_png = png_path, summary_csv = summary_path))