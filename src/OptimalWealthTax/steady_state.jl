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

function steady_state_vector(steady::SteadyStateResult)
    return [steady.k, steady.c, steady.q, steady.Λ1, steady.Λ2, steady.Λ3]
end

function orthonormal_real_basis(columns::AbstractMatrix{<:Real}; tol::Float64 = 1e-9)
    if size(columns, 2) == 0
        return zeros(size(columns, 1), 0)
    end

    factorization = svd(Matrix{Float64}(columns))
    if isempty(factorization.S)
        return zeros(size(columns, 1), 0)
    end

    cutoff = max(tol, factorization.S[1] * tol)
    rank = count(σ -> σ > cutoff, factorization.S)
    return rank == 0 ? zeros(size(columns, 1), 0) : Matrix(factorization.U[:, 1:rank])
end

function real_invariant_basis(values::AbstractVector, vectors::AbstractMatrix; select::Function, tol::Float64 = 1e-9)
    basis_columns = Vector{Vector{Float64}}()
    used = falses(length(values))

    for i in eachindex(values)
        if used[i] || !select(values[i])
            continue
        end

        λ = values[i]
        v = vectors[:, i]
        if abs(imag(λ)) <= tol
            push!(basis_columns, collect(real.(v)))
            used[i] = true
            continue
        end

        if imag(λ) > tol
            push!(basis_columns, collect(real.(v)))
            push!(basis_columns, collect(imag.(v)))
        end

        used[i] = true
        for j in i + 1:length(values)
            if used[j] || !select(values[j])
                continue
            end
            if abs(real(values[j]) - real(λ)) <= 100 * tol && abs(imag(values[j]) + imag(λ)) <= 100 * tol
                used[j] = true
                break
            end
        end
    end

    raw_basis = isempty(basis_columns) ? zeros(length(values), 0) : hcat(basis_columns...)
    return orthonormal_real_basis(raw_basis; tol = tol)
end

"""
        steady_state_linearization(p=ModelParams(); tol=1e-9)

Builds the Jacobian of the six-dimensional optimal-tax dynamics at the exact steady state and
reports the local stable/unstable splitting.

Input arguments:
- `p::ModelParams = ModelParams()`: parameter set.

Optional parameters:
- `tol::Float64 = 1e-9`: real-part tolerance used to classify eigenvalues.

Output:
- Returns a named tuple with fields `steady`, `J`, `eigenvalues`, `classification`, `counts`,
    `stable_basis`, `unstable_basis`, `center_basis`, `center_stable_basis`, `stable_complement_basis`,
    and `center_stable_complement_basis`.
- `J` is a `6 x 6` matrix, each basis is a dense real matrix with 6 rows, and the basis column
    counts report the real dimensions of the corresponding invariant subspaces.
"""
function steady_state_linearization(p::ModelParams = ModelParams(); tol::Float64 = 1e-9)
    steady = find_steady_state(p)
    y_star = steady_state_vector(steady)
    J = ForwardDiff.jacobian(y -> dynamics(y, p), y_star)
    eig = eigen(J)
    values = eig.values
    vectors = eig.vectors

    stable = count(λ -> real(λ) < -tol, values)
    unstable = count(λ -> real(λ) > tol, values)
    center = length(values) - stable - unstable
    classification = unstable > 0 && stable > 0 ? "saddle" :
        (unstable == 0 && stable == length(values) ? "locally asymptotically stable" :
        (stable == 0 && unstable > 0 ? "unstable" : "center/degenerate"))

    stable_basis = real_invariant_basis(values, vectors; select = λ -> real(λ) < -tol, tol = tol)
    unstable_basis = real_invariant_basis(values, vectors; select = λ -> real(λ) > tol, tol = tol)
    center_basis = real_invariant_basis(values, vectors; select = λ -> abs(real(λ)) <= tol, tol = tol)
    center_stable_basis = orthonormal_real_basis(hcat(stable_basis, center_basis); tol = tol)
    stable_complement_basis = nullspace(Matrix(stable_basis'); atol = tol, rtol = tol)
    center_stable_complement_basis = nullspace(Matrix(center_stable_basis'); atol = tol, rtol = tol)

    return (
        steady = steady,
        J = J,
        eigenvalues = values,
        classification = classification,
        counts = (stable = stable, unstable = unstable, center = center),
        dimensions = (
            stable = size(stable_basis, 2),
            unstable = size(unstable_basis, 2),
            center = size(center_basis, 2),
            stable_complement = size(stable_complement_basis, 2),
            center_stable = size(center_stable_basis, 2),
            center_stable_complement = size(center_stable_complement_basis, 2),
        ),
        stable_basis = stable_basis,
        unstable_basis = unstable_basis,
        center_basis = center_basis,
        center_stable_basis = center_stable_basis,
        stable_complement_basis = Matrix(stable_complement_basis),
        center_stable_complement_basis = Matrix(center_stable_complement_basis),
    )
end