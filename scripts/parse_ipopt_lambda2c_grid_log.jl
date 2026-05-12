function parse_start(line)
    match_obj = match(r"=== START T=(\d+) N=(\d+) iter=(\d+) (.+) ===", line)
    return match_obj === nothing ? nothing : (
        T = parse(Int, match_obj.captures[1]),
        N = parse(Int, match_obj.captures[2]),
        iter = parse(Int, match_obj.captures[3]),
        started_at = match_obj.captures[4],
    )
end

function parse_end(line)
    match_obj = match(r"=== END T=(\d+) N=(\d+) status=(\d+) (.+) ===", line)
    return match_obj === nothing ? nothing : (
        T = parse(Int, match_obj.captures[1]),
        N = parse(Int, match_obj.captures[2]),
        status = parse(Int, match_obj.captures[3]),
        ended_at = match_obj.captures[4],
    )
end

function main()
    outputs_dir = joinpath(@__DIR__, "..", "outputs")
    log_paths = sort(filter(isfile, [
        joinpath(outputs_dir, "ipopt_lambda2c_grid.log"),
        joinpath(outputs_dir, "ipopt_lambda2c_grid_T100.log"),
        joinpath(outputs_dir, "ipopt_lambda2c_grid_T200.log"),
    ]))
    if isempty(log_paths)
        println((cases = 0, completed = 0, running = 0, logs = String[]))
        return
    end

    cases = Dict{Tuple{Int,Int}, Dict{String,Any}}()
    for log_path in log_paths
        for line in eachline(log_path)
            start_info = parse_start(line)
            if start_info !== nothing
                cases[(start_info.T, start_info.N)] = Dict(
                    "T" => start_info.T,
                    "N" => start_info.N,
                    "iter" => start_info.iter,
                    "started_at" => start_info.started_at,
                    "status" => missing,
                    "ended_at" => missing,
                )
                continue
            end
            end_info = parse_end(line)
            if end_info !== nothing
                case = get!(cases, (end_info.T, end_info.N), Dict{String,Any}())
                case["T"] = end_info.T
                case["N"] = end_info.N
                case["status"] = end_info.status
                case["ended_at"] = end_info.ended_at
            end
        end
    end

    rows = collect(values(cases))
    sort!(rows, by = row -> (row["T"], row["N"]))
    completed = count(row -> !ismissing(row["status"]), rows)
    running = length(rows) - completed

    for row in rows
        println((T = row["T"], N = row["N"], iter = get(row, "iter", missing), status = row["status"], started_at = row["started_at"], ended_at = row["ended_at"]))
    end
    println((cases = length(rows), completed = completed, running = running, logs = log_paths))
end

main()