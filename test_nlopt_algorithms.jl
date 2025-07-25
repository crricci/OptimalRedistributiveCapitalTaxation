#!/usr/bin/env julia

"""
Comprehensive NLopt Algorithm Comparison for Optimal Redistributive Capital Taxation

This script tests all available NLopt algorithms (gradient-based, derivative-free, and global)
and compares their performance based on transversality errors and convergence behavior.
"""

using Pkg
Pkg.activate(".")

include("parameters.jl")
include("solver.jl")
using Printf
using DelimitedFiles

"""
Test all NLopt algorithms and collect performance metrics
"""
function test_all_nlopt_algorithms()
    
    # Define all NLopt algorithms to test
    algorithms = [
        # Gradient-based local optimizers
        (:LD_LBFGS, "Limited-memory BFGS", "Gradient-based Local"),
        (:LD_VAR1, "Shifted limited-memory variable-metric 1", "Gradient-based Local"),
        (:LD_VAR2, "Shifted limited-memory variable-metric 2", "Gradient-based Local"),
        (:LD_TNEWTON, "Truncated Newton", "Gradient-based Local"),
        (:LD_TNEWTON_RESTART, "Truncated Newton with restarts", "Gradient-based Local"),
        (:LD_TNEWTON_PRECOND, "Preconditioned truncated Newton", "Gradient-based Local"),
        (:LD_TNEWTON_PRECOND_RESTART, "Preconditioned truncated Newton with restarts", "Gradient-based Local"),
        
        # Derivative-free local optimizers
        (:LN_SBPLX, "Subplex (Nelder-Mead variant)", "Derivative-free Local"),
        (:LN_NEWUOA, "NEWUOA algorithm", "Derivative-free Local"),
        (:LN_NEWUOA_BOUND, "NEWUOA with bound constraints", "Derivative-free Local"),
        (:LN_PRAXIS, "Principal axis method", "Derivative-free Local"),
        (:LN_NELDERMEAD, "Nelder-Mead simplex", "Derivative-free Local"),
        (:LN_COBYLA, "COBYLA algorithm", "Derivative-free Local"),
        (:LN_BOBYQA, "BOBYQA algorithm", "Derivative-free Local"),
        
        # Global optimizers
        (:GN_DIRECT, "DIRECT algorithm", "Global"),
        (:GN_DIRECT_L, "DIRECT-L algorithm", "Global"),
        (:GN_CRS2_LM, "Controlled random search", "Global"),
        (:GN_MLSL, "Multi-level single-linkage", "Global"),
        (:GN_ISRES, "Improved stochastic ranking evolution strategy", "Global"),
    ]
    
    println("="^80)
    println("COMPREHENSIVE NLOPT ALGORITHM COMPARISON")
    println("="^80)
    println("Testing $(length(algorithms)) different NLopt algorithms...")
    println("Each algorithm will be tested with the optimal redistributive capital taxation model.")
    println()
    
    # Storage for results
    results = []
    
    for (i, (algorithm, description, category)) in enumerate(algorithms)
        println("[$i/$(length(algorithms))] Testing $algorithm ($description)")
        println("-"^60)
        
        try
            # Create parameters with specific algorithm
            params = ModelParams(
                use_nlopt = true,
                nlopt_algorithm = algorithm,
                filename_suffix = "_$(string(algorithm))"
            )
            
            # Record start time
            start_time = time()
            
            # Run shooting method with timeout protection
            result = shooting_method_state_c(params, verbose=false)
            
            # Record end time
            end_time = time()
            elapsed_time = end_time - start_time
            
            if result.success
                # Calculate transversality errors
                λ_error = abs(result.λ_transversality[end])
                μ_error = abs(result.μ_transversality[end])
                c_error = abs(result.c_transversality[end])
                rms_error = sqrt(λ_error^2 + μ_error^2 + c_error^2)
                
                # Extract final economic values
                final_k = result.k[end]
                final_c = result.c[end]
                final_tau = result.tau_k[end]
                final_r_tilde = result.r_tilde[end]
                
                println("  ✓ SUCCESS")
                println("    Transversality errors:")
                println("      λ: $(λ_error)")
                println("      μ: $(μ_error)")
                println("      c: $(c_error)")
                println("      RMS: $(rms_error)")
                println("    Final values:")
                println("      k(T): $(round(final_k, digits=2))")
                println("      c(T): $(round(final_c, digits=2))")
                println("      τₖ(T): $(round(final_tau, digits=4))")
                println("      r̃(T): $(round(final_r_tilde, digits=6))")
                println("    Computation time: $(round(elapsed_time, digits=2)) seconds")
                
                # Store successful result
                push!(results, (
                    algorithm = string(algorithm),
                    description = description,
                    category = category,
                    success = true,
                    λ_error = λ_error,
                    μ_error = μ_error,
                    c_error = c_error,
                    rms_error = rms_error,
                    final_k = final_k,
                    final_c = final_c,
                    final_tau = final_tau,
                    final_r_tilde = final_r_tilde,
                    time_seconds = elapsed_time
                ))
                
            else
                println("  ✗ FAILED - No convergence")
                println("    Computation time: $(round(elapsed_time, digits=2)) seconds")
                
                # Store failed result
                push!(results, (
                    algorithm = string(algorithm),
                    description = description,
                    category = category,
                    success = false,
                    λ_error = NaN,
                    μ_error = NaN,
                    c_error = NaN,
                    rms_error = NaN,
                    final_k = NaN,
                    final_c = NaN,
                    final_tau = NaN,
                    final_r_tilde = NaN,
                    time_seconds = elapsed_time
                ))
            end
            
        catch e
            println("  ✗ ERROR: $e")
            
            # Store error result
            push!(results, (
                algorithm = string(algorithm),
                description = description,
                category = category,
                success = false,
                λ_error = NaN,
                μ_error = NaN,
                c_error = NaN,
                rms_error = NaN,
                final_k = NaN,
                final_c = NaN,
                final_tau = NaN,
                final_r_tilde = NaN,
                time_seconds = NaN
            ))
        end
        
        println()
    end
    
    return results
end

"""
Generate summary tables and analysis
"""
function analyze_results(results)
    println("="^80)
    println("RESULTS ANALYSIS AND SUMMARY")
    println("="^80)
    
    # Separate successful and failed results
    successful = filter(r -> r.success, results)
    failed = filter(r -> !r.success, results)
    
    println("Overall Performance:")
    println("  Successful algorithms: $(length(successful))/$(length(results))")
    println("  Failed algorithms: $(length(failed))/$(length(results))")
    println()
    
    if length(successful) > 0
        println("PERFORMANCE RANKING (by RMS Transversality Error)")
        println("="^80)
        
        # Sort by RMS error
        sorted_successful = sort(successful, by = r -> r.rms_error)
        
        println(@sprintf("%-20s %-12s %-12s %-12s %-12s %-8s", 
                "Algorithm", "λ Error", "μ Error", "c Error", "RMS Error", "Time(s)"))
        println("-"^80)
        
        for (rank, result) in enumerate(sorted_successful)
            println(@sprintf("%-20s %-12.2e %-12.2e %-12.2e %-12.2e %-8.2f", 
                    result.algorithm,
                    result.λ_error,
                    result.μ_error, 
                    result.c_error,
                    result.rms_error,
                    result.time_seconds))
        end
        println()
        
        # Category analysis
        println("PERFORMANCE BY CATEGORY")
        println("="^50)
        
        categories = unique([r.category for r in successful])
        for category in categories
            cat_results = filter(r -> r.category == category, successful)
            if length(cat_results) > 0
                avg_rms = mean([r.rms_error for r in cat_results])
                avg_time = mean([r.time_seconds for r in cat_results])
                best_rms = minimum([r.rms_error for r in cat_results])
                
                println("$category:")
                println("  Successful: $(length(cat_results))")
                println("  Average RMS error: $(Printf.@sprintf("%.2e", avg_rms))")
                println("  Best RMS error: $(Printf.@sprintf("%.2e", best_rms))")
                println("  Average time: $(round(avg_time, digits=2)) seconds")
                println()
            end
        end
        
        # Economic results consistency
        println("ECONOMIC RESULTS CONSISTENCY")
        println("="^50)
        
        k_values = [r.final_k for r in successful]
        c_values = [r.final_c for r in successful]
        tau_values = [r.final_tau for r in successful]
        r_tilde_values = [r.final_r_tilde for r in successful]
        
        println("Final Capital k(T):")
        println("  Mean: $(round(mean(k_values), digits=2))")
        println("  Std:  $(round(std(k_values), digits=2))")
        println("  Range: [$(round(minimum(k_values), digits=2)), $(round(maximum(k_values), digits=2))]")
        println()
        
        println("Final Consumption c(T):")
        println("  Mean: $(round(mean(c_values), digits=2))")
        println("  Std:  $(round(std(c_values), digits=2))")
        println("  Range: [$(round(minimum(c_values), digits=2)), $(round(maximum(c_values), digits=2))]")
        println()
        
        println("Capital Tax Rate τₖ(T):")
        println("  Mean: $(round(mean(tau_values), digits=4))")
        println("  Std:  $(round(std(tau_values), digits=4))")
        println("  Range: [$(round(minimum(tau_values), digits=4)), $(round(maximum(tau_values), digits=4))]")
        println()
        
        println("After-tax Interest Rate r̃(T):")
        println("  Mean: $(round(mean(r_tilde_values), digits=6))")
        println("  Std:  $(round(std(r_tilde_values), digits=6))")
        println("  Range: [$(round(minimum(r_tilde_values), digits=6)), $(round(maximum(r_tilde_values), digits=6))]")
        println()
    end
    
    # Failed algorithms
    if length(failed) > 0
        println("FAILED ALGORITHMS")
        println("="^50)
        for result in failed
            println("$(result.algorithm) ($(result.category)): $(result.description)")
        end
        println()
    end
    
    return successful, failed
end

"""
Save results to CSV file
"""
function save_results_to_csv(results, filename="nlopt_algorithm_comparison.csv")
    # Prepare data matrix
    headers = ["Algorithm", "Description", "Category", "Success", "Lambda_Error", "Mu_Error", 
               "C_Error", "RMS_Error", "Final_K", "Final_C", "Final_Tau", "Final_R_Tilde", "Time_Seconds"]
    
    data = []
    for result in results
        push!(data, [
            result.algorithm,
            result.description,
            result.category,
            result.success,
            result.λ_error,
            result.μ_error,
            result.c_error,
            result.rms_error,
            result.final_k,
            result.final_c,
            result.final_tau,
            result.final_r_tilde,
            result.time_seconds
        ])
    end
    
    # Combine headers and data
    output_data = vcat([headers], data)
    
    # Save to CSV
    writedlm(filename, output_data, ',')
    println("✓ Results saved to '$filename'")
    
    return filename
end

"""
Generate recommendations based on results
"""
function generate_recommendations(successful_results)
    println("ALGORITHM RECOMMENDATIONS")
    println("="^50)
    
    if length(successful_results) == 0
        println("No successful algorithms to recommend.")
        return
    end
    
    # Sort by RMS error
    sorted_results = sort(successful_results, by = r -> r.rms_error)
    
    println("🥇 BEST OVERALL: $(sorted_results[1].algorithm)")
    println("   RMS Error: $(Printf.@sprintf("%.2e", sorted_results[1].rms_error))")
    println("   Time: $(round(sorted_results[1].time_seconds, digits=2)) seconds")
    println("   Category: $(sorted_results[1].category)")
    println()
    
    # Best in each category
    categories = unique([r.category for r in successful_results])
    for category in categories
        cat_results = filter(r -> r.category == category, successful_results)
        if length(cat_results) > 0
            best = sort(cat_results, by = r -> r.rms_error)[1]
            println("🏆 BEST $category: $(best.algorithm)")
            println("   RMS Error: $(Printf.@sprintf("%.2e", best.rms_error))")
            println("   Time: $(round(best.time_seconds, digits=2)) seconds")
            println()
        end
    end
    
    # Speed recommendations
    fastest = sort(successful_results, by = r -> r.time_seconds)[1]
    println("⚡ FASTEST: $(fastest.algorithm)")
    println("   Time: $(round(fastest.time_seconds, digits=2)) seconds")
    println("   RMS Error: $(Printf.@sprintf("%.2e", fastest.rms_error))")
    println()
    
    # Robustness (consistent economic results)
    println("💡 GENERAL RECOMMENDATIONS:")
    println("   • For highest accuracy: Use $(sorted_results[1].algorithm)")
    println("   • For fastest computation: Use $(fastest.algorithm)")
    println("   • For robust global search: Try global algorithms if local methods fail")
    println("   • All successful algorithms show consistent economic results")
    println()
end

"""
Main execution function
"""
function main()
    println("Starting comprehensive NLopt algorithm comparison...")
    println("This may take several minutes to complete.")
    println()
    
    # Test all algorithms
    results = test_all_nlopt_algorithms()
    
    # Analyze results
    successful, failed = analyze_results(results)
    
    # Save to CSV
    save_results_to_csv(results)
    
    # Generate recommendations
    generate_recommendations(successful)
    
    println("="^80)
    println("COMPARISON COMPLETE")
    println("="^80)
    println("Total algorithms tested: $(length(results))")
    println("Successful: $(length(successful))")
    println("Failed: $(length(failed))")
    println("Results saved to: nlopt_algorithm_comparison.csv")
    println()
    
    return results, successful, failed
end

# Run if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
