# Gamma Sensitivity Analysis
# Optimal Redistributive Capital Taxation Model (Consumption as State Variable)

# Activate the project environment
using Pkg
Pkg.activate(@__DIR__)

using PyPlot
using Statistics

include("parameters.jl")
include("solver.jl")

function gamma_sensitivity_state()
    """
    Gamma sensitivity analysis with values from 0.01 to 5
    """
    println("=" ^ 70)
    println("GAMMA SENSITIVITY ANALYSIS")
    println("=" ^ 70)
    
    # Gamma values from 0.01 to 5
    gamma_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    # Storage for results
    results = Dict(
        "gamma" => Float64[],
        "final_k" => Float64[],
        "final_c" => Float64[],
        "final_lambda" => Float64[],
        "final_mu" => Float64[],
        "final_tau_k" => Float64[],
        "final_r_tilde" => Float64[],
        "lambda_transversality_error" => Float64[],
        "mu_transversality_error" => Float64[],
        "success" => Bool[]
    )
    
    println("Testing gamma values from 0.01 to 5: $gamma_values")
    
    # Test analysis
    for gamma in gamma_values
        println("\n--- Testing γ = $gamma ---")
        
        # Create parameters with current gamma
        params = ModelParams(γ=gamma, T=1000.0)
        
        try
            # Solve using shooting method
            result = shooting_method_state_c(params, save_points=2000, verbose=false)
            
            if result.success
                # Extract values at T/2 instead of final time
                T_max = maximum(result.t)
                T_half = T_max / 2
                half_idx = findmin(abs.(result.t .- T_half))[2]  # Find closest index to T/2
                
                println("  ✓ Success: r̃=$(round(result.r_tilde[half_idx], digits=6)), τₖ=$(round(result.tau_k[half_idx], digits=6)) (at T/2)")
                
                push!(results["gamma"], gamma)
                push!(results["final_k"], result.k[half_idx])
                push!(results["final_c"], result.c[half_idx])
                push!(results["final_lambda"], result.λ[half_idx])
                push!(results["final_mu"], result.μ[half_idx])
                push!(results["final_tau_k"], result.tau_k[half_idx])
                push!(results["final_r_tilde"], result.r_tilde[half_idx])
                push!(results["lambda_transversality_error"], abs(result.λ_transversality[end]))  # Keep transversality at end
                push!(results["mu_transversality_error"], abs(result.μ_transversality[end]))      # Keep transversality at end
                push!(results["success"], true)
            else
                println("  ✗ Failed to converge")
                push!(results["gamma"], gamma)
                push!(results["final_k"], NaN)
                push!(results["final_c"], NaN)
                push!(results["final_lambda"], NaN)
                push!(results["final_mu"], NaN)
                push!(results["final_tau_k"], NaN)
                push!(results["final_r_tilde"], NaN)
                push!(results["lambda_transversality_error"], NaN)
                push!(results["mu_transversality_error"], NaN)
                push!(results["success"], false)
            end
        catch e
            println("  ✗ Error: $e")
            push!(results["gamma"], gamma)
            push!(results["final_k"], NaN)
            push!(results["final_c"], NaN)
            push!(results["final_lambda"], NaN)
            push!(results["final_mu"], NaN)
            push!(results["final_tau_k"], NaN)
            push!(results["final_r_tilde"], NaN)
            push!(results["lambda_transversality_error"], NaN)
            push!(results["mu_transversality_error"], NaN)
            push!(results["success"], false)
        end
    end
    
    # Generate plot
    plot_gamma_sensitivity_state(results)
    
    println("\n✓ Gamma sensitivity completed!")
    println("  Check 'gamma_sensitivity_state.png' for results")
    
    return results
end

function plot_gamma_sensitivity_state(state_results)
    """
    Generate visualization for gamma sensitivity analysis
    Values are extracted at T/2 for analysis purposes.
    """
    # Filter successful results
    successful_mask = state_results["success"]
    gamma_vals = state_results["gamma"][successful_mask]
    tau_k_vals = state_results["final_tau_k"][successful_mask]
    r_tilde_vals = state_results["final_r_tilde"][successful_mask]
    k_vals = state_results["final_k"][successful_mask]
    c_vals = state_results["final_c"][successful_mask]
    
    if length(gamma_vals) == 0
        println("No successful results to plot")
        return
    end
    
    # Create plot with only tax rate and interest rate
    figure(figsize=(12, 5))
    
    # Plot 1: Tax Rate vs Gamma
    subplot(1, 2, 1)
    plot(gamma_vals, tau_k_vals * 100, "b-o", markersize=6, linewidth=2)
    xlabel("γ (Redistribution Cost)", fontsize=12)
    ylabel("Tax Rate τₖ (%)", fontsize=12)
    title("Optimal Tax Rate vs Redistribution Cost", fontsize=14, fontweight="bold")
    grid(true, alpha=0.3)
    ylim(75, 85)
    
    # Plot 2: Interest Rate vs Gamma
    subplot(1, 2, 2)
    plot(gamma_vals, r_tilde_vals * 100, "r-s", markersize=6, linewidth=2)
    xlabel("γ (Redistribution Cost)", fontsize=12)
    ylabel("Effective Interest Rate r̃ (%)", fontsize=12)
    title("Effective Interest Rate vs Redistribution Cost", fontsize=14, fontweight="bold")
    grid(true, alpha=0.3)
    
    # Overall title
    suptitle("Gamma Sensitivity Analysis - Tax Rate and Interest Rate", 
             fontsize=16, fontweight="bold")
    
    # Adjust layout and save
    tight_layout()
    savefig("gamma_sensitivity_state.png", dpi=300, bbox_inches="tight")
    println("Sensitivity plot saved as 'gamma_sensitivity_state.png'")
    
    # Print summary statistics
    println("\nSummary Statistics:")
    println("  γ range: $(minimum(gamma_vals)) - $(maximum(gamma_vals))")
    println("  τₖ range: $(round(minimum(tau_k_vals)*100, digits=2))% - $(round(maximum(tau_k_vals)*100, digits=2))%")
    println("  r̃ range: $(round(minimum(r_tilde_vals)*100, digits=4))% - $(round(maximum(r_tilde_vals)*100, digits=4))%")
    println("  Successful convergence: $(sum(successful_mask))/$(length(successful_mask)) cases")
    
    # Reference case (γ = 0.5)
    ref_idx = findfirst(x -> abs(x - 0.5) < 1e-6, gamma_vals)
    if ref_idx !== nothing
        println("Reference case: γ = $(gamma_vals[ref_idx]), r̃ = $(round(r_tilde_vals[ref_idx]*100, digits=6))%, τₖ = $(round(tau_k_vals[ref_idx]*100, digits=6))%")
    end
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    gamma_sensitivity_state()
end
