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

    case_name = "ipopt_tvc_T$(Int(round(T)))_N$(N)_beta$(replace(string(beta), "." => "p"))"
    output_dir = joinpath(@__DIR__, "..", "outputs", case_name)

    run = _solveOptimalWealthTaxation(
        output_dir = output_dir,
        progress = false,
        model_kwargs = (; T = T, N = N, max_iter = max_iter, β = beta),
        solve_kwargs = (; terminal_mode = :tvc, use_nested_seed = true, use_horizon_continuation = false),
    )

    tvc = OptimalWealthTax.transversality_metrics(run.result, run.params).terminal
    println((
        case = case_name,
        success = run.result.success,
        residual = run.result.residual_norm,
        terminal_time = run.result.t[end],
        final_k = run.result.k[end],
        final_q = run.result.q[end],
        tvc_k = tvc.k,
        tvc_c = tvc.c,
        tvc_q = tvc.q,
        output_dir = output_dir,
        solution_csv = run.solution_csv,
        plot_png = run.plot_png,
        summary_csv = run.summary_csv,
    ))
end

main(ARGS)