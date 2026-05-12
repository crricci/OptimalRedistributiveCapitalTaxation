include(joinpath(@__DIR__, "..", "main.jl"))

function read_summary_row(path)
    lines = readlines(path)
    if length(lines) < 2
        return nothing
    end
    headers = split(lines[1], ',')
    values = split(lines[2], ',')
    return Dict(headers[i] => values[i] for i in eachindex(headers))
end

function main()
    outputs_dir = joinpath(@__DIR__, "..", "outputs")
    case_dirs = sort(filter(name -> startswith(name, "ipopt_tvc_lambda2c_T") && isdir(joinpath(outputs_dir, name)), readdir(outputs_dir)))
    rows = Vector{Dict{String,String}}()

    for case_dir in case_dirs
        summary_path = joinpath(outputs_dir, case_dir, "OptimalWealthTaxation_summary.csv")
        if !isfile(summary_path)
            continue
        end
        row = read_summary_row(summary_path)
        row === nothing && continue
        T = parse(Float64, get(row, "horizon_T", "0"))
        T >= 50.0 || continue
        row["case_dir"] = case_dir
        push!(rows, row)
    end

    output_path = joinpath(outputs_dir, "ipopt_lambda2c_grid_beta075_summary.csv")
    headers = [
        "case_dir",
        "success",
        "residual_norm",
        "terminal_mode",
        "terminal_mid_label",
        "horizon_T",
        "mesh_N",
        "final_k",
        "steady_k",
        "final_c",
        "steady_c",
        "final_q",
        "steady_q",
        "terminal_tvc_k",
        "terminal_tvc_c",
        "terminal_tvc_q",
        "solution_csv",
        "plot_png",
    ]
    write_csv_table(output_path, headers, ([get(row, header, "") for header in headers] for row in rows))
    println((summary_csv = output_path, cases = length(rows)))
end

main()