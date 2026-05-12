function read_summary(path)
    lines = isfile(path) ? readlines(path) : String[]
    length(lines) >= 2 || return nothing
    headers = split(lines[1], ',')
    values = split(lines[2], ',')
    return Dict(headers[i] => values[i] for i in eachindex(headers))
end

function collect_summaries(paths)
    rows = Dict{Tuple{Int,Int}, Vector{Dict{String,String}}}()
    for path in paths
        row = read_summary(path)
        row === nothing && continue
        T = round(Int, parse(Float64, row["horizon_T"]))
        N = parse(Int, row["mesh_N"])
        key = (T, N)
        push!(get!(rows, key, Dict{String,String}[]), merge(row, Dict("summary_path" => path)))
    end
    return rows
end

function main()
    outputs_dir = joinpath(@__DIR__, "..", "outputs")
    ipopt_paths = filter(isfile, [joinpath(outputs_dir, d, "OptimalWealthTaxation_summary.csv") for d in readdir(outputs_dir) if startswith(d, "ipopt_tvc_T")])
    baseline_paths = filter(path -> !occursin("ipopt_tvc_", path), [joinpath(root, file) for (root, _, files) in walkdir(outputs_dir) for file in files if file == "OptimalWealthTaxation_summary.csv"])

    ipopt_rows = collect_summaries(ipopt_paths)
    baseline_rows = collect_summaries(baseline_paths)

    lines = String[]
    push!(lines, "# Ipopt vs Baselines")
    push!(lines, "")
    push!(lines, "| T | N | ipopt_residual | baseline_residual | baseline_path |")
    push!(lines, "| --- | --- | --- | --- | --- |")
    for key in sort(collect(keys(ipopt_rows)))
        T, N = key
        T >= 50 || continue
        ipopt_best = sort(ipopt_rows[key], by = row -> parse(Float64, row["residual_norm"]))[1]
        baselines = get(baseline_rows, key, Dict{String,String}[])
        if isempty(baselines)
            push!(lines, "| $(T) | $(N) | $(ipopt_best["residual_norm"]) |  |  |")
            continue
        end
        baseline_best = sort(baselines, by = row -> parse(Float64, row["residual_norm"]))[1]
        push!(lines, "| $(T) | $(N) | $(ipopt_best["residual_norm"]) | $(baseline_best["residual_norm"]) | $(baseline_best["summary_path"]) |")
    end

    report_path = joinpath(outputs_dir, "ipopt_vs_baselines.md")
    write(report_path, join(lines, '\n'))
    println((report = report_path, rows = length(lines) - 4))
end

main()