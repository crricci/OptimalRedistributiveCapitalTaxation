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
    r::Float64 = 0.05     # Interest rate
    k0::Float64 = 2.0     # Initial capital
    T::Float64 = 100.0   # Time horizon for solution
end

