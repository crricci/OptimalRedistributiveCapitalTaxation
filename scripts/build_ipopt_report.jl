function parse_csv(path)
    lines = isfile(path) ? readlines(path) : String[]
    isempty(lines) && return String[], Vector{Dict{String,String}}()
    headers = split(lines[1], ',')
    rows = Dict{String,String}[]
    for line in Iterators.drop(lines, 1)
        isempty(line) && continue
        values = split(line, ',')
        push!(rows, Dict(headers[i] => values[i] for i in eachindex(headers)))
    end
    return headers, rows
end

function parse_statuses(path)
    statuses = Dict{Tuple{Int,Int},Int}()
    if !isfile(path)
        return statuses
    end
    for line in eachline(path)
        match_obj = match(r"=== END T=(\d+) N=(\d+) status=(\d+) .+ ===", line)
        match_obj === nothing && continue
        statuses[(parse(Int, match_obj.captures[1]), parse(Int, match_obj.captures[2]))] = parse(Int, match_obj.captures[3])
    end
    return statuses
end

function parse_all_statuses(outputs_dir)
    statuses = Dict{Tuple{Int,Int},Int}()
    for path in filter(isfile, [
        joinpath(outputs_dir, "ipopt_grid_beta075.log"),
        joinpath(outputs_dir, "ipopt_grid_beta075_T100.log"),
        joinpath(outputs_dir, "ipopt_grid_beta075_T200.log"),
        joinpath(outputs_dir, "ipopt_single_T50_N41.log"),
        joinpath(outputs_dir, "ipopt_single_T100_N41.log"),
    ])
        merge!(statuses, parse_statuses(path))
    end
    return statuses
end

function main()
    outputs_dir = joinpath(@__DIR__, "..", "outputs")
    summary_csv = joinpath(outputs_dir, "ipopt_grid_beta075_summary.csv")
    _, rows = parse_csv(summary_csv)
    statuses = parse_all_statuses(outputs_dir)
    rows = filter(row -> parse(Float64, get(row, "horizon_T", "0")) >= 50.0, rows)

    sort!(rows, by = row -> (parse(Int, split(row["horizon_T"], '.')[1]), parse(Int, row["mesh_N"])))

    report_lines = String[]
    push!(report_lines, "# Ipopt Grid Report")
    push!(report_lines, "")
    push!(report_lines, "| T | N | success | residual | status | final_k | final_q | tvc_k | tvc_c | tvc_q | plot |")
    push!(report_lines, "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows
        T = parse(Int, split(row["horizon_T"], '.')[1])
        N = parse(Int, row["mesh_N"])
        status = get(statuses, (T, N), -999)
        plot_path = row["plot_png"]
        push!(report_lines,
            "| $(T) | $(N) | $(row["success"]) | $(row["residual_norm"]) | $(status) | $(row["final_k"]) | $(row["final_q"]) | $(row["terminal_tvc_k"]) | $(row["terminal_tvc_c"]) | $(row["terminal_tvc_q"]) | $(plot_path) |")
    end

    report_path = joinpath(outputs_dir, "ipopt_grid_beta075_report.md")
    write(report_path, join(report_lines, '\n'))
    println((report = report_path, rows = length(rows)))
end

main()