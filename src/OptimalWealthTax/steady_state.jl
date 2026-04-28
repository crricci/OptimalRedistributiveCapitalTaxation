"""
    SteadyStateResult

Struct storing the steady state of the `OptimalWealthTax` model.

Fields:
- `k`, `c`, `q`, `Λ1`, `Λ2`, `Λ3`, `r_tilde`, `x`: all are `Float64` scalars.

Dimensions:
- Each field has size `1 x 1`; this struct contains no time trajectories.
"""
struct SteadyStateResult
    k::Float64
    c::Float64
    q::Float64
    Λ1::Float64
    Λ2::Float64
    Λ3::Float64
    r_tilde::Float64
    x::Float64
end

"""
    find_steady_state()
    find_steady_state(p)

Computes the steady state of the `OptimalWealthTax` model.

Input arguments:
- None for the zero-argument method, which uses `ModelParams()`.
- `p::ModelParams` for the parameterized method.

Optional parameters:
- None beyond the fields already stored in `p`.

Output:
- Returns a `SteadyStateResult`.
- All components are scalars with size `1 x 1`.
"""
find_steady_state() = find_steady_state(ModelParams())

function find_steady_state(p::ModelParams)
    scale = production_scale(p)
    base = (p.ρ + p.δ) / (scale * p.θ)
    if !(isfinite(base) && base > p.min_positive)
        error("Invalid steady-state base for k*: $(base)")
    end

    k = base^(1.0 / (p.θ - 1.0))
    terms = production_terms(k, p)
    q = terms.Fn / p.ρ
    c = p.ρ * k + terms.Fn
    x = terms.F - (p.δ + p.ρ) * k - terms.Fn
    if !(isfinite(x) && x > p.min_positive)
        error("Invalid steady-state x*: $(x)")
    end

    denom = c / p.β - p.ρ * (k + q)
    if abs(denom) <= p.min_positive
        error("Singular steady-state denominator in Λ2 formula")
    end

    Λ2 = ((p.γ * (k + q) / x) - c^(-p.β) * (k + q)) / denom
    Λ1 = c^(-p.β) - p.ρ * Λ2
    Λ3 = Λ1 - p.γ / x

    return SteadyStateResult(k, c, q, Λ1, Λ2, Λ3, p.ρ, x)
end