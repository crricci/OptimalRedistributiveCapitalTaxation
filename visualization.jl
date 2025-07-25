# Visualization functions for Optimal Redistributive Capital Taxation Model

using PyPlot
using Statistics
using DelimitedFiles

"""
    plot_main_solution(result, title, filename)

Create the main solution plot with vertical layout (8 plots) matching the uploaded image layout.
Only saves to PNG, does not display.
"""
function plot_main_solution(result, title, filename)
    if !result.success
        println("Cannot visualize - solution failed to converge")
        return
    end
    
    # Only plot data up to T/2 (first half of the solution)
    T_max = maximum(result.t)
    T_half = T_max / 2
    half_indices = result.t .<= T_half
    
    t_plot = result.t[half_indices]
    k_plot = result.k[half_indices]
    c_plot = result.c[half_indices]
    λ_plot = result.λ[half_indices]
    μ_plot = result.μ[half_indices]
    r_tilde_plot = result.r_tilde[half_indices]
    tau_k_plot = result.tau_k[half_indices]
    λ_transversality_plot = result.λ_transversality[half_indices]
    μ_transversality_plot = result.μ_transversality[half_indices]
    
    # Create 8 subplots in vertical layout (8 rows, 1 column)
    fig, ax = PyPlot.subplots(8, 1, figsize=(12, 16))
    fig.suptitle(title * " (0 to T/2)", fontsize=18)
    
    # Capital Trajectory (c as State) - Blue line
    ax[1].plot(t_plot, k_plot, "b-", linewidth=2)
    ax[1].set_title("Capital Trajectory (c as State)")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Capital k(t)")
    ax[1].grid(true)
    
    # Consumption Trajectory (State Variable) - Purple/magenta line
    ax[2].plot(t_plot, c_plot, "m-", linewidth=2)
    ax[2].set_title("Consumption Trajectory (State Variable)")
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("Consumption c(t)")
    ax[2].grid(true)
    
    # Costate λ Trajectory - Red line
    ax[3].plot(t_plot, λ_plot, "r-", linewidth=2)
    ax[3].set_title("Costate λ Trajectory")
    ax[3].set_xlabel("Time")
    ax[3].set_ylabel("Costate λ(t)")
    ax[3].grid(true)
    
    # Costate μ Trajectory - Orange line
    ax[4].plot(t_plot, μ_plot, color="orange", linewidth=2)
    ax[4].set_title("Costate μ Trajectory")
    ax[4].set_xlabel("Time")
    ax[4].set_ylabel("Costate μ(t)")
    ax[4].grid(true)
    
    # Effective Interest Rate - Green line
    ax[5].plot(t_plot, r_tilde_plot, "g-", linewidth=2)
    ax[5].set_title("Effective Interest Rate")
    ax[5].set_xlabel("Time")
    ax[5].set_ylabel("Effective Interest Rate")
    ax[5].grid(true)
    
    # Capital Tax Rate - Cyan line
    ax[6].plot(t_plot, tau_k_plot, "c-", linewidth=2)
    ax[6].set_title("Capital Tax Rate")
    ax[6].set_xlabel("Time")
    ax[6].set_ylabel("Capital Tax Rate")
    ax[6].grid(true)
    
    # λ Transversality Condition - Magenta line
    ax[7].plot(t_plot, λ_transversality_plot, "m-", linewidth=2)
    ax[7].set_title("λ Transversality Condition")
    ax[7].set_xlabel("Time")
    ax[7].set_ylabel("λ Transversality")
    ax[7].grid(true)
    
    # μ Transversality Condition - Red line
    ax[8].plot(t_plot, μ_transversality_plot, "r-", linewidth=2)
    ax[8].set_title("μ Transversality Condition")
    ax[8].set_xlabel("Time")
    ax[8].set_ylabel("μ Transversality")
    ax[8].grid(true)
    
    PyPlot.tight_layout()
    PyPlot.savefig(filename, dpi=300, bbox_inches="tight")
    PyPlot.close(fig)  # Close the figure to prevent display
    println("✓ Plot saved as '$filename'")
end

"""
    plot_welfare_analysis(welfare_results, filename_base="welfare_analysis_gamma", params=nothing)

Generate visualization for welfare analysis as function of gamma.
Creates a single welfare plot showing the welfare integral as a function of gamma.
"""
function plot_welfare_analysis(welfare_results, filename_base="welfare_analysis_gamma", params=nothing)
    # Filter successful results
    successful_mask = welfare_results["success"]
    gamma_vals = welfare_results["gamma"][successful_mask]
    welfare_vals = welfare_results["welfare_integral"][successful_mask]
    
    if length(gamma_vals) == 0
        println("No successful results to plot")
        return
    end
    
    # Create single welfare plot
    figure(figsize=(12, 8))
    
    # Main welfare plot
    plot(gamma_vals, welfare_vals, "b-o", markersize=8, linewidth=3, markerfacecolor="lightblue", markeredgecolor="blue")
    xlabel("Redistribution Cost Parameter γ", fontsize=14)
    ylabel("Welfare Integral", fontsize=14)
    title("\$\\int_{0}^{\\infty} [\\gamma V(x) + U(c)]e^{-\\rho t} dt\$", fontsize=18, fontweight="bold")
    grid(true, alpha=0.3)
    
    # Find optimal gamma (but don't annotate it)
    max_welfare_idx = argmax(welfare_vals)
    optimal_gamma = gamma_vals[max_welfare_idx]
    max_welfare = welfare_vals[max_welfare_idx]
    
    # Construct filename with suffix if provided
    if params !== nothing && !isempty(params.filename_suffix)
        filename = filename_base * params.filename_suffix * ".png"
    else
        filename = filename_base * ".png"
    end
    
    # Adjust layout and save
    tight_layout()
    savefig(filename, dpi=300, bbox_inches="tight")
    println("Welfare analysis plot saved as '$filename'")
    
    # Print detailed summary
    println("\n" * "="^60)
    println("DETAILED WELFARE ANALYSIS RESULTS")
    println("="^60)
    println("Welfare Function: ∫₀^∞ [γV(x) + U(c)]e^(-ρt) dt")
    println("where V(x) = log(x) and U(c) = (c^(1-β) - 1)/(1-β)")
    println()
    
    println("Optimal Results:")
    println("  γ* = $optimal_gamma (optimal redistribution cost parameter)")
    println("  Maximum Welfare = $(round(max_welfare, digits=6))")
    
    # Get final values for optimal gamma (at T/2)
    final_c = welfare_results["final_consumption"][successful_mask]
    final_x = welfare_results["final_redistribution"][successful_mask]
    println("  Final Consumption c(T/2) = $(round(final_c[max_welfare_idx], digits=4))")
    println("  Final Redistribution x(T/2) = $(round(final_x[max_welfare_idx], digits=4))")
    
    # Compare with reference cases
    println()
    println("Comparison with Reference Cases:")
    
    # γ = 0.5 case
    ref_05_idx = findfirst(x -> abs(x - 0.5) < 1e-6, gamma_vals)
    if ref_05_idx !== nothing
        ref_welfare = welfare_vals[ref_05_idx]
        improvement = ((max_welfare - ref_welfare) / abs(ref_welfare)) * 100
        println("  γ = 0.5: Welfare = $(round(ref_welfare, digits=6)) ($(round(improvement, digits=2))% improvement from optimal)")
    end
    
    # γ = 1.0 case
    ref_10_idx = findfirst(x -> abs(x - 1.0) < 1e-6, gamma_vals)
    if ref_10_idx !== nothing
        ref_welfare = welfare_vals[ref_10_idx]
        improvement = ((max_welfare - ref_welfare) / abs(ref_welfare)) * 100
        println("  γ = 1.0: Welfare = $(round(ref_welfare, digits=6)) ($(round(improvement, digits=2))% improvement from optimal)")
    end
    
    println()
    println("Range Statistics:")
    println("  γ range (successful): $(minimum(gamma_vals)) - $(maximum(gamma_vals))")
    println("  Welfare range: $(round(minimum(welfare_vals), digits=4)) - $(round(maximum(welfare_vals), digits=4))")
    println("  Successful cases: $(sum(successful_mask))/$(length(welfare_results["gamma"]))")
end

"""
    load_welfare_results(filename="welfare_analysis_results.csv")

Load welfare analysis results from CSV file.
"""
function load_welfare_results(filename="welfare_analysis_results.csv")
    if !isfile(filename)
        error("File '$filename' not found. Run welfare analysis first to generate the data.")
    end
    
    # Read CSV file
    data_matrix = readdlm(filename, ',')
    
    # Extract header and data
    header = data_matrix[1, :]
    data = data_matrix[2:end, :]
    
    # Convert to dictionary format
    results = Dict(
        "gamma" => Float64.(data[:, 1]),
        "welfare_integral" => Float64.(data[:, 2]),
        "success" => Bool.(data[:, 3]),
        "final_consumption" => Float64.(data[:, 4]),
        "final_redistribution" => Float64.(data[:, 5]),
        "avg_consumption_utility" => Float64.(data[:, 6]),
        "avg_redistribution_utility" => Float64.(data[:, 7])
    )
    
    println("✓ Welfare results loaded from '$filename'")
    return results
end

"""
    plot_welfare_from_file(filename="welfare_analysis_results.csv", params=nothing)

Load welfare analysis results from CSV file and generate visualization plot.
"""
function plot_welfare_from_file(filename="welfare_analysis_results.csv", params=nothing)
    println("Loading welfare analysis results from file...")
    welfare_results = load_welfare_results(filename)
    
    println("Generating welfare plot...")
    plot_welfare_analysis(welfare_results, "welfare_analysis_gamma", params)
    
    return welfare_results
end

"""
    plot_welfare_quick()

Quick function to plot welfare results from saved CSV file.
This replaces the functionality of the separate plot_welfare.jl file.
"""
function plot_welfare_quick(filename="welfare_analysis_results.csv", params=nothing)
    println("Plotting welfare analysis from saved results...")
    welfare_results = plot_welfare_from_file(filename, params)
    return welfare_results
end
