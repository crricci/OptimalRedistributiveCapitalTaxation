# Solving methods for the Optimal Redistributive Capital Taxation Model

using DifferentialEquations
using Optim
using ForwardDiff
using NLsolve
using NLopt
using LinearAlgebra  # For norm function
using Statistics     # For mean function
using DelimitedFiles # For CSV saving

include("parameters.jl")

# ========================================
# CONSUMPTION AS STATE VARIABLE SYSTEM  
# ========================================

# System of ODEs for the formulation where c is a state variable
# State variables: k, c
# Costate variables: λ, μ  
# System: [k, c, λ, μ]
function system_ode_state_c!(du, u, params, t)
    k, c, λ, μ = u[1], u[2], u[3], u[4]
    
    # Ensure strictly positive values with tighter bounds
    k = max(k, 1e-6)  # Lower bound for capital
    c = max(c, 1e-6)  # Lower bound for consumption
    λ = max(λ, 1e-6)  # Lower bound for costate λ
    μ = max(μ, 1e-6)  # Lower bound for costate μ
    
    # Add upper bounds to prevent explosive growth
    k = min(k, 1e6)
    c = min(c, 1e6)
    λ = min(λ, 1e6)
    μ = min(μ, 1e6)
    
    # Compute constraint term: λ + μc/(βk) = γ/x
    constraint_lhs = λ + μ * c / (params.β * k)
    
    # Ensure constraint_lhs is bounded away from zero
    constraint_lhs = max(constraint_lhs, 1e-6)
    
    # This gives us: x = γ/constraint_lhs
    x = params.γ / constraint_lhs
    
    # From x = A(1-η)k^θ - (δ + r̃)k, solve for r̃
    # x = A(1-η)k^θ - (δ + r̃)k
    # r̃ = [A(1-η)k^θ - x]/k - δ
    term1 = params.A * (1 - params.η) * k^params.θ
    r_unconstrained = (term1 - x) / k - params.δ
    
    # Smooth approximation of max(0, r_unconstrained) using parameter α_state
    r_tilde = log(1.0 + exp(params.α_state * r_unconstrained)) / params.α_state
    
    # Bound r_tilde to reasonable economic values
    r_tilde = max(0.0, min(0.5, r_tilde))  # Interest rate between 0% and 50%
    
    # State equations
    # dk/dt = r̃k + Aηk^θ - c
    production = params.A * params.η * k^params.θ
    dk_dt = r_tilde * k + production - c
    
    # dc/dt = (c/β)(r̃ - ρ)
    dc_dt = (c / params.β) * (r_tilde - params.ρ)
    
    # Costate equations
    # dλ/dt = λ(ρ - r̃ - Aθηk^(θ-1)) - (γ/x)(Aθ(1-η)k^(θ-1) - δ - r̃)
    marginal_prod_capital = params.A * params.θ * params.η * k^(params.θ - 1)
    marginal_x_k = params.A * params.θ * (1 - params.η) * k^(params.θ - 1)
    
    dλ_dt = λ * (params.ρ - r_tilde - marginal_prod_capital) - 
            (params.γ / x) * (marginal_x_k - params.δ - r_tilde)
    
    # dμ/dt = μ(ρ - (r̃ - ρ)/β)
    dμ_dt = μ * (params.ρ - (r_tilde - params.ρ) / params.β)
    
    # Prevent extreme values to maintain numerical stability
    max_rate = 50.0  # Maximum allowed rate of change
    dk_dt = max(-max_rate, min(max_rate, dk_dt))
    dc_dt = max(-max_rate, min(max_rate, dc_dt))
    dλ_dt = max(-max_rate, min(max_rate, dλ_dt))
    dμ_dt = max(-max_rate, min(max_rate, dμ_dt))
    
    du[1] = dk_dt
    du[2] = dc_dt
    du[3] = dλ_dt
    du[4] = dμ_dt
    
    return nothing
end

# Solve the system with consumption as state variable using transversality condition
function solve_system_state_c(params; λ0=1.0, μ0=1.0, c0=nothing, T=nothing, save_points=1000, verbose=false)
    if T === nothing
        T = params.T
    end
    
    # If c0 is not provided, we'll determine it through shooting method
    if c0 === nothing
        c0 = 1.0  # Initial guess, will be optimized
    end
    
    # Initial conditions: [k(0), c(0), λ(0), μ(0)]
    u0 = [params.k0, c0, λ0, μ0]
    tspan = (0.0, T)
    
    # Create problem
    prob = ODEProblem(system_ode_state_c!, u0, tspan, params)
    
    # Solve with robust method suited for stiff problems
    sol = solve(prob, RadauIIA5(), 
                abstol=1e-8, reltol=1e-6,
                saveat=range(0, T, length=save_points),
                maxiters=1e6,
                dtmin=1e-12, dtmax=1.0,
                force_dtmin=true)
    
    if string(sol.retcode) == "Success"
        if verbose
            println("ODE solver succeeded, checking solution validity...")
        end
        
        t_vals = sol.t
        k_vals = [u[1] for u in sol.u]
        c_vals = [u[2] for u in sol.u]
        λ_vals = [u[3] for u in sol.u]
        μ_vals = [u[4] for u in sol.u]
        
        if verbose
            println("Solution extracted, checking for validity...")
            println("  Number of time points: $(length(t_vals))")
            println("  k range: [$(minimum(k_vals)), $(maximum(k_vals))]")
            println("  c range: [$(minimum(c_vals)), $(maximum(c_vals))]")
        end
        
        # Check for invalid values
        has_nan = any(isnan.(k_vals)) || any(isnan.(c_vals)) || any(isnan.(λ_vals)) || any(isnan.(μ_vals))
        has_inf = any(isinf.(k_vals)) || any(isinf.(c_vals)) || any(isinf.(λ_vals)) || any(isinf.(μ_vals))
        has_nonpos = any(k_vals .<= 0) || any(c_vals .<= 0) || any(λ_vals .<= 0) || any(μ_vals .<= 0)
        
        if has_nan || has_inf || has_nonpos
            if verbose
                println("Solution contains invalid values:")
                println("  NaN values: $has_nan")
                println("  Inf values: $has_inf") 
                println("  Non-positive values: $has_nonpos")
                if has_nonpos
                    println("  k min: $(minimum(k_vals)), c min: $(minimum(c_vals))")
                    println("  λ min: $(minimum(λ_vals)), μ min: $(minimum(μ_vals))")
                end
            end
            return (success=false, t=Float64[], k=Float64[], c=Float64[], λ=Float64[], μ=Float64[],
                    r_tilde=Float64[], tau_k=Float64[], x=Float64[],
                    λ_transversality=Float64[], μ_transversality=Float64[])
        end
        
        # Compute derived quantities
        r_tilde_vals = Float64[]
        tau_k_vals = Float64[]
        x_vals = Float64[]
        
        for (k, c, λ, μ) in zip(k_vals, c_vals, λ_vals, μ_vals)
            # Compute x
            x = params.γ / (λ + μ * c / (params.β * k))
            push!(x_vals, x)
            
            # Compute r̃
            term1 = params.A * (1 - params.η) * k^params.θ
            r_unconstrained = (term1 - x) / k - params.δ
            r_tilde = log(1.0 + exp(params.α_state * r_unconstrained)) / params.α_state
            push!(r_tilde_vals, r_tilde)
            
            # Compute capital tax rate τₖ from r̃ = r(1 - τₖ)
            if params.r > 0
                tau_k = 1.0 - r_tilde / params.r
                tau_k = max(0.0, min(1.0, tau_k))  # Clamp between 0 and 1
            else
                tau_k = 0.0
            end
            push!(tau_k_vals, tau_k)
        end
        
        # Compute transversality conditions
        # λ transversality: e^(-ρt) * λ(t) * k(t) → 0
        λ_transversality = [exp(-params.ρ * t) * λ * k for (t, λ, k) in zip(t_vals, λ_vals, k_vals)]
        
        # μ transversality: e^(-ρt) * μ(t) * c(t) → 0  
        μ_transversality = [exp(-params.ρ * t) * μ * c for (t, μ, c) in zip(t_vals, μ_vals, c_vals)]
        
        # Consumption transversality: c(t)^(-β) * k(t) * e^(-ρt) → 0
        c_transversality = [exp(-params.ρ * t) * (c^(-params.β)) * k for (t, c, k) in zip(t_vals, c_vals, k_vals)]
        
        return (success=true, t=t_vals, k=k_vals, c=c_vals, λ=λ_vals, μ=μ_vals,
                r_tilde=r_tilde_vals, tau_k=tau_k_vals, x=x_vals,
                λ_transversality=λ_transversality, μ_transversality=μ_transversality, 
                c_transversality=c_transversality)
    else
        println("ODE solver failed with return code: $(sol.retcode)")
        if verbose
            println("Solver details: retcode = $(sol.retcode)")
        end
        return (success=false, t=Float64[], k=Float64[], c=Float64[], λ=Float64[], μ=Float64[],
                r_tilde=Float64[], tau_k=Float64[], x=Float64[],
                λ_transversality=Float64[], μ_transversality=Float64[], c_transversality=Float64[])
    end
end

# Advanced shooting method using Optim or NLopt for 4-variable system
function shooting_method_state_c(params; T=nothing, save_points=1000, verbose=false, use_nlopt=nothing)
    if T === nothing
        T = params.T
    end
    
    # Use parameter from structure if not explicitly overridden
    if use_nlopt === nothing
        use_nlopt = params.use_nlopt
    end
    
    optimizer_name = use_nlopt ? "NLopt ($(params.nlopt_algorithm))" : "Optim"
    if verbose
        println("Using $optimizer_name shooting method for 4-variable system...")
    end
    
    # Objective function: minimize weighted sum of transversality errors
    function objective_func(x)
        # x = [c0, λ0, μ0] - we optimize initial consumption and costate variables
        # k0 is fixed from parameters
        
        if length(x) != 3
            return 1e10
        end
        
        c0, λ0, μ0 = x[1], x[2], x[3]
        
        # Ensure positive values
        if c0 <= 0 || λ0 <= 0 || μ0 <= 0
            return 1e10
        end
        
        # Solve the ODE system
        result = solve_system_state_c(params, λ0=λ0, μ0=μ0, c0=c0, T=T, save_points=save_points, verbose=false)
        
        if !result.success
            return 1e10
        end
        
        # Check for economic validity
        if any(result.k .<= 0) || any(result.c .<= 0)
            return 1e10
        end
        
        # Compute transversality errors at final time (already properly scaled)
        # Note: transversality conditions already include exp(-ρt) factor
        
        # Three transversality conditions to satisfy
        λ_error = abs(result.λ_transversality[end])
        μ_error = abs(result.μ_transversality[end])  
        c_error = abs(result.c_transversality[end])
        
        # Weighted objective (equal weights)
        total_error = λ_error + μ_error + c_error
        
        if verbose && total_error < 1e-6
            println("  Trial: c0=$(round(c0,digits=4)), λ0=$(round(λ0,digits=4)), μ0=$(round(μ0,digits=4))")
            println("    Errors: λ=$(round(λ_error,sigdigits=3)), μ=$(round(μ_error,sigdigits=3)), c=$(round(c_error,sigdigits=3))")
        end
        
        return total_error
    end
    
    # Try multiple starting points for robustness
    best_result = nothing
    best_error = Inf
    
    starting_points = [
        [1.0, 2.0, 1.0],    # Standard guess
        [0.5, 1.5, 0.8],    # Lower values
        [2.0, 3.0, 1.5],    # Higher values  
        [1.5, 2.5, 1.2],    # Mid-range values
        [0.8, 1.8, 0.9]     # Alternative values
    ]
    
    for (i, x0) in enumerate(starting_points)
        if verbose
            println("Starting point $i: c0=$(x0[1]), λ0=$(x0[2]), μ0=$(x0[3])")
        end
        
        try
            if use_nlopt
                # Use configurable NLopt algorithm with automatic differentiation
                algorithm = params.nlopt_algorithm
                opt = NLopt.Opt(algorithm, 3)  # 3 variables: c0, λ0, μ0
                
                # Set bounds (all variables must be positive)
                NLopt.lower_bounds!(opt, [1e-6, 1e-6, 1e-6])
                NLopt.upper_bounds!(opt, [1e6, 1e6, 1e6])
                
                # Check if algorithm requires gradients
                algorithm_str = string(algorithm)
                needs_gradients = startswith(algorithm_str, "LD_")
                
                if needs_gradients
                    # Define objective function with gradient computation using ForwardDiff
                    function objective_with_grad!(x, grad)
                        # Compute objective value
                        f_val = objective_func(x)
                        
                        # Compute gradient using automatic differentiation if grad is provided
                        if length(grad) > 0
                            try
                                grad_computed = ForwardDiff.gradient(objective_func, x)
                                grad[1] = grad_computed[1]
                                grad[2] = grad_computed[2] 
                                grad[3] = grad_computed[3]
                            catch e
                                # If gradient computation fails, set to zero (fallback)
                                grad[1] = 0.0
                                grad[2] = 0.0
                                grad[3] = 0.0
                            end
                        end
                        
                        return f_val
                    end
                    
                    # Set objective function with gradient
                    NLopt.min_objective!(opt, objective_with_grad!)
                else
                    # For derivative-free algorithms, use simple objective
                    NLopt.min_objective!(opt, (x, grad) -> objective_func(x))
                end
                
                # Set tolerances
                NLopt.ftol_rel!(opt, 1e-8)
                NLopt.xtol_rel!(opt, 1e-6)
                NLopt.maxeval!(opt, 1000)
                
                # Optimize
                (minf, minx, ret) = NLopt.optimize(opt, x0)
                
                converged = (ret == :FTOL_REACHED || ret == :XTOL_REACHED || ret == :SUCCESS)
                
                if verbose
                    println("  Result: converged=$converged, return code=$ret, error = $(round(minf, sigdigits=4)) [with AD]")
                end
                
            else
                # Use Optim with BFGS optimizer
                # Adjusted tolerances for properly scaled transversality conditions
                result_opt = Optim.optimize(objective_func, x0, BFGS(), 
                                    Optim.Options(g_tol=1e-6, f_reltol=1e-8, iterations=200, show_trace=false))
                
                minf = result_opt.minimum
                minx = result_opt.minimizer
                converged = Optim.converged(result_opt)
                
                if verbose
                    println("  Result: converged=$converged, error = $(round(minf, sigdigits=4))")
                end
            end
            
            if (converged || minf < 1e-3) && minf < best_error
                best_error = minf
                
                # Solve with optimal parameters
                c0_opt, λ0_opt, μ0_opt = minx[1], minx[2], minx[3]
                best_result = solve_system_state_c(params, λ0=λ0_opt, μ0=μ0_opt, c0=c0_opt, 
                                                  T=T, save_points=save_points, verbose=false)
                
                if verbose
                    println("  ✓ New best solution found!")
                    println("    Optimal: c0=$(round(c0_opt,digits=4)), λ0=$(round(λ0_opt,digits=4)), μ0=$(round(μ0_opt,digits=4))")
                    println("    Final error: $(round(best_error, sigdigits=4))")
                end
            end
        catch e
            if verbose
                println("  ✗ Optimization failed: $e")
            end
        end
    end
    
    if best_result !== nothing && best_result.success
        if verbose
            println("✓ $optimizer_name shooting method succeeded!")
            println("  Final transversality errors:")
            println("    λ: $(abs(best_result.λ_transversality[end]))")
            println("    μ: $(abs(best_result.μ_transversality[end]))") 
            println("    c: $(abs(best_result.c_transversality[end]))")
        end
        return best_result
    else
        if verbose
            println("✗ $optimizer_name shooting method failed to find valid solution")
        end
        return create_failed_result()
    end
end

# Helper function for failed results
function create_failed_result()
    return (success=false, t=Float64[], k=Float64[], c=Float64[], λ=Float64[], μ=Float64[],
            r_tilde=Float64[], tau_k=Float64[], x=Float64[],
            λ_transversality=Float64[], μ_transversality=Float64[], c_transversality=Float64[])
end

# Welfare analysis functions
function utility_U(c, β)
    """
    Utility function U(c) = (c^(1-β) - 1)/(1-β)
    """
    if β ≈ 1.0
        return log(c)  # Limiting case when β → 1
    else
        return (c^(1 - β) - 1) / (1 - β)
    end
end

function utility_V(x)
    """
    Redistribution utility function V(x) = log(x)
    """
    return log(max(x, 1e-8))  # Ensure x > 0 for log
end

function compute_welfare_integral(result, params)
    """
    Compute the welfare integral: ∫₀^(T/2) [γV(x) + U(c)]e^(-ρt) dt
    where V(x) = log(x) and U(c) = (c^(1-β) - 1)/(1-β)
    
    Uses trapezoidal rule for numerical integration.
    Note: Only integrates up to T/2 for analysis purposes.
    """
    if !result.success
        return NaN
    end
    
    # Only use data up to T/2 for welfare computation
    T_max = maximum(result.t)
    T_half = T_max / 2
    half_indices = result.t .<= T_half
    
    t_vals = result.t[half_indices]
    c_vals = result.c[half_indices]
    x_vals = result.x[half_indices]
    
    # Compute integrand at each time point
    integrand = Float64[]
    for (t, c, x) in zip(t_vals, c_vals, x_vals)
        # Utility components
        U_c = utility_U(c, params.β)
        V_x = utility_V(x)
        
        # Discounted welfare at time t
        welfare_t = (params.γ * V_x + U_c) * exp(-params.ρ * t)
        push!(integrand, welfare_t)
    end
    
    # Numerical integration using trapezoidal rule
    integral = 0.0
    for i in 2:length(t_vals)
        dt = t_vals[i] - t_vals[i-1]
        integral += 0.5 * (integrand[i] + integrand[i-1]) * dt
    end
    
    return integral
end

function welfare_analysis_gamma(gamma_values=nothing; T=1000.0, save_points=2000, verbose=false, use_nlopt=nothing, params_template=nothing)
    """
    Compute welfare integral as a function of gamma values.
    
    Parameters:
    - gamma_values: Array of gamma values to test (default: same as gamma_sensitivity)
    - T: Time horizon 
    - save_points: Number of points for ODE solution
    - verbose: Print detailed output
    - use_nlopt: Use NLopt instead of Optim for optimization (default: use params.use_nlopt)
    - params_template: Template parameters (default: ModelParams())
    
    Returns:
    - Dictionary with gamma values and corresponding welfare integrals
    """
    
    if params_template === nothing
        params_template = ModelParams()
    end
    
    # Use parameter from structure if not explicitly overridden
    if use_nlopt === nothing
        use_nlopt = params_template.use_nlopt
    end
    
    if gamma_values === nothing
        # Extended range up to gamma = 10
        gamma_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    end
    
    optimizer_name = use_nlopt ? "NLopt ($(params_template.nlopt_algorithm))" : "Optim"
    
    println("=" ^ 70)
    println("WELFARE ANALYSIS AS FUNCTION OF GAMMA")
    println("=" ^ 70)
    println("Computing welfare integral: ∫₀^(T/2) [γV(x) + U(c)]e^(-ρt) dt")
    println("where V(x) = log(x) and U(c) = (c^(1-β) - 1)/(1-β)")
    println("Note: Integration only up to T/2 for analysis purposes")
    println("Using optimizer: $optimizer_name")
    println()
    
    # Storage for results
    results = Dict(
        "gamma" => Float64[],
        "welfare_integral" => Float64[],
        "success" => Bool[],
        "final_consumption" => Float64[],
        "final_redistribution" => Float64[],
        "avg_consumption_utility" => Float64[],
        "avg_redistribution_utility" => Float64[]
    )
    
    println("Testing gamma values: $gamma_values")
    println()
    
    for (idx, gamma) in enumerate(gamma_values)
        if verbose || idx % 5 == 1  # Print progress every 5 iterations
            println("--- Testing γ = $gamma ($(idx)/$(length(gamma_values))) ---")
        end
        
        # Create parameters with current gamma, inheriting other settings from template
        params = ModelParams(params_template; γ=gamma, T=T)
        
        try
            # Solve using shooting method
            result = shooting_method_state_c(params, save_points=save_points, verbose=false, use_nlopt=use_nlopt)
            
            if result.success
                # Compute welfare integral
                welfare_integral = compute_welfare_integral(result, params)
                
                # Extract values at T/2 instead of final time
                T_max = maximum(result.t)
                T_half = T_max / 2
                half_idx = findmin(abs.(result.t .- T_half))[2]  # Find closest index to T/2
                
                # Compute additional statistics using T/2 data
                half_indices = result.t .<= T_half
                avg_U_c = mean([utility_U(c, params.β) for c in result.c[half_indices]])
                avg_V_x = mean([utility_V(x) for x in result.x[half_indices]])
                
                if verbose || idx % 5 == 1
                    println("  ✓ Success: Welfare = $(round(welfare_integral, digits=4))")
                    println("    Final c = $(round(result.c[half_idx], digits=4)), Final x = $(round(result.x[half_idx], digits=4)) (at T/2)")
                end
                
                push!(results["gamma"], gamma)
                push!(results["welfare_integral"], welfare_integral)
                push!(results["success"], true)
                push!(results["final_consumption"], result.c[half_idx])
                push!(results["final_redistribution"], result.x[half_idx])
                push!(results["avg_consumption_utility"], avg_U_c)
                push!(results["avg_redistribution_utility"], avg_V_x)
                
            else
                if verbose || idx % 5 == 1
                    println("  ✗ Failed to converge")
                end
                push!(results["gamma"], gamma)
                push!(results["welfare_integral"], NaN)
                push!(results["success"], false)
                push!(results["final_consumption"], NaN)
                push!(results["final_redistribution"], NaN)
                push!(results["avg_consumption_utility"], NaN)
                push!(results["avg_redistribution_utility"], NaN)
            end
            
        catch e
            if verbose || idx % 5 == 1
                println("  ✗ Error: $e")
            end
            push!(results["gamma"], gamma)
            push!(results["welfare_integral"], NaN)
            push!(results["success"], false)
            push!(results["final_consumption"], NaN)
            push!(results["final_redistribution"], NaN)
            push!(results["avg_consumption_utility"], NaN)
            push!(results["avg_redistribution_utility"], NaN)
        end
    end
    
    # Print summary
    successful_cases = sum(results["success"])
    println()
    println("=" ^ 50)
    println("WELFARE ANALYSIS SUMMARY")
    println("=" ^ 50)
    println("Successful convergence: $successful_cases/$(length(gamma_values)) cases")
    
    if successful_cases > 0
        successful_mask = results["success"]
        welfare_vals = results["welfare_integral"][successful_mask]
        gamma_successful = results["gamma"][successful_mask]
        
        println("Gamma range (successful): $(minimum(gamma_successful)) - $(maximum(gamma_successful))")
        println("Welfare range: $(round(minimum(welfare_vals), digits=4)) - $(round(maximum(welfare_vals), digits=4))")
        
        # Find optimal gamma (maximum welfare)
        max_welfare_idx = argmax(welfare_vals)
        optimal_gamma = gamma_successful[max_welfare_idx]
        max_welfare = welfare_vals[max_welfare_idx]
        
        println()
        println("OPTIMAL GAMMA:")
        println("  γ* = $optimal_gamma")
        println("  Maximum welfare = $(round(max_welfare, digits=6))")
        
        # Reference case (γ = 0.5)
        ref_idx = findfirst(x -> abs(x - 0.5) < 1e-6, gamma_successful)
        if ref_idx !== nothing
            ref_welfare = welfare_vals[ref_idx]
            welfare_improvement = ((max_welfare - ref_welfare) / abs(ref_welfare)) * 100
            println("  Improvement over γ=0.5: $(round(welfare_improvement, digits=2))%")
        end
    end
    
    println()
    println("✓ Welfare analysis completed!")
    
    return results
end

function save_welfare_results(results, filename="welfare_analysis_results", params=nothing)
    """
    Save welfare analysis results to CSV file
    
    Parameters:
    - results: Dictionary with welfare analysis results
    - filename: Base filename (without extension)
    - params: ModelParams structure (used for filename suffix)
    """
    
    # Add suffix if provided
    if params !== nothing && params.filename_suffix != ""
        filename = filename * "_" * params.filename_suffix
    end
    
    # Add CSV extension
    full_filename = filename * ".csv"
    # Prepare data matrix
    data = hcat(
        results["gamma"],
        results["welfare_integral"], 
        results["success"],
        results["final_consumption"],
        results["final_redistribution"],
        results["avg_consumption_utility"],
        results["avg_redistribution_utility"]
    )
    
    # Create header row
    header = ["gamma" "welfare_integral" "success" "final_consumption_at_T_half" "final_redistribution_at_T_half" "avg_consumption_utility" "avg_redistribution_utility"]
    
    # Combine header and data
    output_data = vcat(header, data)
    
    # Save to CSV
    writedlm(full_filename, output_data, ',')
    println("✓ Welfare results saved to '$full_filename'")
    
    return full_filename
end
