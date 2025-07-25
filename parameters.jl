# Parameters for the Optimal Redistributive Capital Taxation Model

using Parameters

# Model parameters structure using Parameters.jl
@with_kw struct ModelParams
    A::Float64 = 1.0      # Productivity parameter
    θ::Float64 = 0.3      # Capital elasticity in production
    η::Float64 = 0.6      # Capital share in production
    ρ::Float64 = 0.03     # Discount rate
    β::Float64 = 0.75     # Inverse elasticity of substitution
    δ::Float64 = 0.1      # Depreciation rate
    γ::Float64 = 0.5      # Redistribution cost parameter
    r::Float64 = 0.15     # Interest rate
    k0::Float64 = 4.0     # Initial capital
    T::Float64 = 1000.0   # Time horizon for solution
    α_state::Float64 = 10.0     # Regularization parameter for max(0,x) approximation
    
    # Optimizer and file handling parameters
    use_nlopt::Bool = false                   # Use NLopt instead of Optim (default: false)
    nlopt_algorithm::Symbol = :LN_NELDERMEAD  # NLopt algorithm choice (default: LN_NELDERMEAD)
    filename_suffix::String = ""   # Suffix for output files (default: empty)
end

# Default parameter values
function default_params()
    return ModelParams()
end