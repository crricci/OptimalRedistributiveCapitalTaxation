"""
    write_csv_table(file_path, headers, rows)

Writes a generic table to CSV.

Input arguments:
- `file_path::AbstractString`: output file path.
- `headers::Vector{String}`: header row of length `M`.
- `rows`: iterable of row tuples or row vectors, each expected to have `M` entries.

Optional parameters:
- None.

Output:
- Returns `file_path`.
- Writes one header row and as many data rows as provided by `rows`.
"""
function write_csv_table(file_path::AbstractString, headers::Vector{String}, rows)
    open(file_path, "w") do io
        println(io, join(headers, ","))
        for row in rows
            println(io, join(string.(row), ","))
        end
    end
    return file_path
end

"""
    default_no_wealth_taxation_output_dir()

Returns the default output directory for `NoWealthTaxation` runs.

Input arguments:
- None.

Optional parameters:
- None.

Output:
- Returns an `AbstractString` path.
- The output is a single path string.
"""
default_no_wealth_taxation_output_dir() = joinpath(normpath(joinpath(@__DIR__, "..")), "outputs", "no_wealth_taxation")

"""
    writeNoWealthTaxationResultCSV(result, file_path)

Writes the full `NoWealthTaxation` solution path and diagnostic series to CSV.

Input arguments:
- `result::NoWealthTaxation.SolutionResult`: solution object with trajectory length `N = length(result.t)`.
- `file_path::AbstractString`: output CSV path.

Optional parameters:
- None.

Output:
- Returns `file_path`.
- Writes `N` data rows and 10 columns.
"""
function writeNoWealthTaxationResultCSV(result::NoWealthTaxation.SolutionResult, file_path::AbstractString)
    headers = ["t", "k", "c", "lambda", "mu", "r_tilde", "tau_k", "lambda_tr", "mu_tr", "c_tr"]
    rows = ((result.t[i], result.k[i], result.c[i], result.λ[i], result.μ[i], result.r_tilde[i], result.tau_k[i], result.λ_tr[i], result.μ_tr[i], result.c_tr[i]) for i in eachindex(result.t))
    return write_csv_table(file_path, headers, rows)
end

"""
    _solveNoWealthTaxation(; k0_values=[2.0, 2.2], output_dir=default_no_wealth_taxation_output_dir(), progress=true, model_kwargs=(;), solve_kwargs=(;), run_gamma_scans=true, gamma_scan_k0=NaN, gamma_welfare_scan_kwargs=(;), gamma_steadystate_welfare_scan_kwargs=(;))

Runs the full `NoWealthTaxation` workflow for one or more initial capital values and optional `γ` scans.

Input arguments:
- No positional arguments.

Optional parameters:
- `k0_values::AbstractVector{<:Real}`: vector of initial capital values, length `K`.
- `output_dir::AbstractString`: output directory.
- `progress::Bool = true`: print progress messages.
- `model_kwargs::NamedTuple`: keyword overrides passed to `NoWealthTaxation.ModelParams`.
- `solve_kwargs::NamedTuple`: keyword overrides passed to `NoWealthTaxation.solve_orct`.
- `run_gamma_scans::Bool = true`: whether to run welfare and steady-state welfare scans.
- `gamma_scan_k0::Real = NaN`: initial capital used for the welfare scan; if `NaN`, the first `k0` value is used.
- `gamma_welfare_scan_kwargs::NamedTuple`: extra keywords for `run_gamma_welfare_scan`.
- `gamma_steadystate_welfare_scan_kwargs::NamedTuple`: extra keywords for `run_gamma_steadystate_welfare_scan`.

Output:
- Returns a named tuple with fields `results`, `summary_csv`, and `gamma_outputs`.
- `results` is a vector of length `K`, and each stored solution contains trajectories of length determined by the underlying solver grid.
"""
function _solveNoWealthTaxation(; k0_values::AbstractVector{<:Real} = [2.0, 2.2], output_dir::AbstractString = default_no_wealth_taxation_output_dir(), progress::Bool = true, model_kwargs::NamedTuple = (;), solve_kwargs::NamedTuple = (;), run_gamma_scans::Bool = true, gamma_scan_k0::Real = NaN, gamma_welfare_scan_kwargs::NamedTuple = (;), gamma_steadystate_welfare_scan_kwargs::NamedTuple = (;))
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

    gamma_outputs = nothing
    if run_gamma_scans
        welfare_k0 = isnan(Float64(gamma_scan_k0)) ? Float64(isempty(k0_values) ? 2.0 : first(k0_values)) : Float64(gamma_scan_k0)
        welfare_csv_path = joinpath(output_dir, "welfare_gamma_scan.csv")
        welfare_png_path = joinpath(output_dir, "welfare_vs_gamma.png")
        steadystate_welfare_csv_path = joinpath(output_dir, "gamma_steadystate_welfare_scan.csv")
        steadystate_welfare_png_path = joinpath(output_dir, "gamma_vs_steadystate_welfare.png")

        welfare_scan_kwargs = (; gamma_welfare_scan_kwargs..., k0 = welfare_k0, outfile = welfare_csv_path, progress = progress)
        progress && println("\n=== NoWealthTaxation: gamma welfare scan (k0=$(welfare_k0)) ===")
        run_gamma_welfare_scan(; welfare_scan_kwargs...)
        NoWealthTaxation.plot_welfare_vs_gamma(welfare_csv_path, welfare_png_path)

        steadystate_welfare_scan_kwargs = (; gamma_steadystate_welfare_scan_kwargs..., outfile = steadystate_welfare_csv_path, progress = progress)
        progress && println("\n=== NoWealthTaxation: gamma steady-state welfare scan ===")
        run_gamma_steadystate_welfare_scan(; steadystate_welfare_scan_kwargs...)
        NoWealthTaxation.plot_gamma_vs_steadystate_welfare(steadystate_welfare_csv_path, steadystate_welfare_png_path)

        gamma_outputs = (; welfare_csv = welfare_csv_path, welfare_png = welfare_png_path, steadystate_welfare_csv = steadystate_welfare_csv_path, steadystate_welfare_png = steadystate_welfare_png_path)
    end

    return (; results, summary_csv = summary_path, gamma_outputs)
end