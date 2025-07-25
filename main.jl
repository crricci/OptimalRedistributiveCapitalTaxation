# Main execution script - Normal Precision with Basic Optim
# Optimal Redistributive Capital Taxation Model

using DifferentialEquations
using Optim
using LinearAlgebra

include("parameters.jl")
include("solver.jl")
include("visualization.jl")

"""
    find_solution()

Find the optimal solution using normal precision method with basic Optim BFGS.
Returns the solution result.
"""
function find_solution()
    println("=" ^ 60)
    println("NORMAL PRECISION METHOD (Basic Optim)")
    println("=" ^ 60)

    # Load parameters
    params = default_params()
    println("✓ Model parameters loaded")
    println("  Time horizon: T = $(params.T)")
    println("  Initial capital: k₀ = $(params.k0)")
    println("  Redistribution: η = $(params.η)")

    # Run normal precision method
    println("\nRunning shooting method with basic Optim BFGS...")
    result = shooting_method_state_c(params, verbose=true)

    if result.success
        println("\n✓ Normal precision method succeeded!")
        
        # Display final values
        println("Final values at T=$(params.T):")
        println("  k(T) = $(round(result.k[end], digits=6))")
        println("  c(T) = $(round(result.c[end], digits=6))")
        println("  τₖ(T) = $(round(result.tau_k[end], digits=6))")
        println("  r̃(T) = $(round(result.r_tilde[end], digits=6))")
        
        # Display transversality errors
        println("\nTransversality condition errors:")
        println("  λ: $(abs(result.λ_transversality[end]))")
        println("  μ: $(abs(result.μ_transversality[end]))")
        println("  c: $(abs(result.c_transversality[end]))")
        
        rms_error = sqrt(result.λ_transversality[end]^2 + 
                        result.μ_transversality[end]^2 + 
                        result.c_transversality[end]^2)
        println("  RMS error: $(round(rms_error, sigdigits=6))")
        
        # Create filename with suffix
        filename = "normal_precision_solution"
        if params.filename_suffix != ""
            filename = filename * params.filename_suffix
        end
        filename = filename * ".png"
        
        # Create visualization
        plot_main_solution(result, "Normal Precision: Complete Economic Analysis", filename)
        
    else
        println("✗ Normal precision method failed to converge")
    end

    println("\n" ^ 2)
    println("=" ^ 60)
    println("NORMAL PRECISION ANALYSIS COMPLETE")
    println("=" ^ 60)
    
    return result
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    find_solution()
end
