include(joinpath(@__DIR__, "..", "src", "OptimalWealthTax.jl"))
include(joinpath(@__DIR__, "..", "src", "NoWealthTaxationRun.jl"))
include(joinpath(@__DIR__, "..", "src", "OptimalWealthTaxationRun.jl"))

outdir = joinpath(@__DIR__, "..", "outputs", "optimal_wealth_taxation_smooth")
mkpath(outdir)

p = Main.OptimalWealthTax.ModelParams(T = 40.0, N = 21, max_iter = 1200, mesh_power = 6.0)
seed = Main.OptimalWealthTax.solve_collocation(p;
    progress = false,
    terminal_mode = :state_steady_state,
    use_continuation = true,
    use_nested_seed = false)

p_dense = Main.OptimalWealthTax.ModelParams(T = 40.0, N = 401, max_iter = 1200, mesh_power = 6.0)
t_dense = Main.OptimalWealthTax.collocation_grid(p_dense, p_dense.N)
z_dense = Main.OptimalWealthTax.interpolate_guess(seed.t, Main.OptimalWealthTax.pack_solution(seed), t_dense)
dense_result = Main.OptimalWealthTax.unpack_solution(z_dense, p_dense, seed.steady, t_dense, seed.residual_norm, seed.success)
tvc = Main.OptimalWealthTax.transversality_metrics(dense_result, p_dense)

csv_path = joinpath(outdir, "OptimalWealthTaxation_solution.csv")
png_path = joinpath(outdir, "OptimalWealthTaxation_solution.png")
summary_path = joinpath(outdir, "OptimalWealthTaxation_summary.csv")

writeOptimalWealthTaxationResultCSV(dense_result, p_dense, csv_path)
Main.OptimalWealthTax.plot_solution(dense_result, "OptimalWealthTaxation smooth path", png_path; force = true)
write_csv_table(summary_path,
    ["success", "residual_norm", "terminal_mode", "horizon_T", "mesh_N", "final_k", "steady_k", "final_c", "steady_c", "final_q", "steady_q", "terminal_tvc_k", "terminal_tvc_c", "terminal_tvc_q", "solution_csv", "plot_png"],
    [(dense_result.success, dense_result.residual_norm, "state_steady_state_interpolated", p_dense.T, p_dense.N, dense_result.k[end], dense_result.steady.k, dense_result.c[end], dense_result.steady.c, dense_result.q[end], dense_result.steady.q, tvc.terminal.k, tvc.terminal.c, tvc.terminal.q, csv_path, png_path)])

println((; success = dense_result.success, residual = dense_result.residual_norm, solution_csv = csv_path, plot_png = png_path, summary_csv = summary_path))