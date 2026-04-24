function write_csv_table(file_path::AbstractString, headers::Vector{String}, rows)
    open(file_path, "w") do io
        println(io, join(headers, ","))
        for row in rows
            println(io, join(string.(row), ","))
        end
    end
    return file_path
end

function writeNoWealthTaxationResultCSV(result::NoWealthTaxation.SolutionResult, file_path::AbstractString)
    headers = ["t", "k", "c", "lambda", "mu", "r_tilde", "tau_k", "lambda_tr", "mu_tr", "c_tr"]
    rows = ((result.t[i], result.k[i], result.c[i], result.λ[i], result.μ[i], result.r_tilde[i], result.tau_k[i], result.λ_tr[i], result.μ_tr[i], result.c_tr[i]) for i in eachindex(result.t))
    return write_csv_table(file_path, headers, rows)
end

function _solveNoWealthTaxation(; k0_values::AbstractVector{<:Real} = [2.0, 2.2], output_dir::AbstractString = pwd(), progress::Bool = true, model_kwargs::NamedTuple = (;), solve_kwargs::NamedTuple = (;))
    mkpath(output_dir)
    results = NamedTuple[]
    summary_rows = Vector{NTuple{8, Any}}()

    for k0 in k0_values
        p = NoWealthTaxation.ModelParams(; model_kwargs..., k0 = Float64(k0))
        progress && println("\n=== NoWealthTaxation: k0=$(p.k0) ===")
        result = NoWealthTaxation.solve_orct(p; progress = progress, solve_kwargs...)

        residuals = try
            compute_residuals(p, result)
        catch
            nothing
        end
        foc_max = residuals === nothing ? NaN : maximum(abs.(residuals.foc_res))
        k_max = residuals === nothing ? NaN : maximum(abs.(residuals.eq_k))

        suffix = replace(string(round(p.k0; digits = 4)), "." => "-")
        csv_path = joinpath(output_dir, "NoWealthTaxation_solution_k0=$(suffix).csv")
        png_path = joinpath(output_dir, "NoWealthTaxation_solution_k0=$(suffix).png")
        writeNoWealthTaxationResultCSV(result, csv_path)
        NoWealthTaxation.plot_main_solution(result, "NoWealthTaxation path (k0=$(p.k0))", png_path; force = true)

        push!(results, (; params = p, result = result, csv_path = csv_path, plot_path = png_path, foc_max = foc_max, k_max = k_max))
        push!(summary_rows, (p.k0, result.success, result.k[end], result.c[end], result.r_tilde[end], foc_max, csv_path, png_path))
    end

    summary_path = joinpath(output_dir, "NoWealthTaxation_summary.csv")
    write_csv_table(summary_path, ["k0", "success", "final_k", "final_c", "final_r_tilde", "foc_max", "solution_csv", "plot_png"], summary_rows)
    return (; results, summary_csv = summary_path)
end