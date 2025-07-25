# Welfare Analysis Script
# Optimal Redistributive Capital Taxation Model

# Activate the project environment
using Pkg
Pkg.activate(@__DIR__)

using PyPlot
using Statistics
using DelimitedFiles

include("parameters.jl")
include("solver.jl")
include("visualization.jl")

function run_welfare_analysis(params=nothing)
    """
    Main function to run complete welfare analysis and save results
    """
    if params === nothing
        params = ModelParams()
    end
    
    println("Starting Welfare Analysis...")
    println("Computing: ∫₀^∞ [γV(x) + U(c)]e^(-ρt) dt")
    println("where V(x) = log(x) and U(c) = (c^(1-β) - 1)/(1-β)")
    println()
    
    # Run welfare analysis with extended gamma range
    welfare_results = welfare_analysis_gamma(verbose=false, params_template=params)
    
    # Save results to CSV
    save_welfare_results(welfare_results, "welfare_analysis_results", params)
    
    # Generate visualization
    plot_welfare_analysis(welfare_results, "welfare_analysis_gamma", params)
    
    return welfare_results
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    welfare_results = run_welfare_analysis()
end
