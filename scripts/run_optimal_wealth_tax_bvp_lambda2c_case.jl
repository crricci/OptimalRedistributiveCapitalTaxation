include(joinpath(@__DIR__, "..", "main.jl"))

function parse_arg(args, idx, name)
    if length(args) < idx
        error("Missing required argument: $(name)")
    end
    return args[idx]
end

function main(args)
    T = parse(Float64, parse_arg(args, 1, "T"))
    N = parse(Int, parse_arg(args, 2, "N"))
    max_iter = parse(Int, parse_arg(args, 3, "max_iter"))
    beta = length(args) >= 4 ? parse(Float64, args[4]) : 0.75

    case_name = "bvp_tvc_lambda2c_T$(Int(round(T)))_N$(N)_beta$(replace(string(beta), "." => "p"))"
    output_dir = joinpath(@__DIR__, "..", "outputs", case_name)

    # Build model parameters
    p = OptimalWealthTax.ModelParams(T = T, N = N, max_iter = max_iter, β = beta)
    steady = OptimalWealthTax.find_steady_state(p)

    # Solve with shooting/BVP method and new terminal condition
    result, guess = OptimalWealthTax.solve_shooting_stage(p, steady;
        progress = true,
        terminal_mode = :tvc_lambda2c,
        output_N = N)

    # Save results (reuse collocation output format)
    mkpath(output_dir)
    solution_csv = joinpath(output_dir, "OptimalWealthTaxation_solution.csv")
    plot_png = joinpath(output_dir, "OptimalWealthTaxation_solution.png")
    summary_csv = joinpath(output_dir, "OptimalWealthTaxation_summary.csv")
    OptimalWealthTaxationRun.writeOptimalWealthTaxationResultCSV(result, p, solution_csv)
    OptimalWealthTaxationRun.writeOptimalWealthTaxationSummaryCSV(result, p, summary_csv)
    OptimalWealthTax.visualization.plot_solution(result, "BVP solution", plot_png)

    tvc = OptimalWealthTax.transversality_metrics(result, p; terminal_mode = :tvc_lambda2c).terminal
    println((
        case = case_name,
        success = result.success,
        residual = result.residual_norm,
        terminal_time = result.t[end],
        final_k = result.k[end],
        final_q = result.q[end],
        tvc_k = tvc.k,
        tvc_c = tvc.c,
        tvc_q = tvc.q,
        output_dir = output_dir,
        solution_csv = solution_csv,
        plot_png = plot_png,
        summary_csv = summary_csv,
    ))
end

main(ARGS)
