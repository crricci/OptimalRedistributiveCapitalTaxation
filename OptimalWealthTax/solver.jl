struct CollocationResult
    success::Bool
    t::Vector{Float64}
    k::Vector{Float64}
    c::Vector{Float64}
    q::Vector{Float64}
    Λ1::Vector{Float64}
    Λ2::Vector{Float64}
    Λ3::Vector{Float64}
    r_tilde::Vector{Float64}
    x::Vector{Float64}
    steady::SteadyStateResult
    residual_norm::Float64
end

node_offset(i::Int) = 6 * (i - 1)

function node_slice(z::AbstractVector, i::Int)
    offset = node_offset(i)
    return @view z[offset + 1:offset + 6]
end

function collocation_guess(p::ModelParams, steady::SteadyStateResult, tgrid::AbstractVector)
    N = length(tgrid)
    guess = zeros(6 * N)
    for (i, t) in enumerate(tgrid)
        w = tgrid[end] <= 0 ? 0.0 : t / tgrid[end]
        offset = node_offset(i)
        guess[offset + 1] = (1.0 - w) * p.k0 + w * steady.k
        guess[offset + 2] = (1.0 - w) * steady.c + w * steady.c
        guess[offset + 3] = (1.0 - w) * p.q0 + w * steady.q
        guess[offset + 4] = steady.Λ1
        guess[offset + 5] = (1.0 - w) * 0.0 + w * steady.Λ2
        guess[offset + 6] = steady.Λ3
    end
    return guess
end

function interpolate_guess(old_t::AbstractVector, old_z::AbstractVector, new_t::AbstractVector)
    old_n = length(old_t)
    new_n = length(new_t)
    new_z = zeros(6 * new_n)
    for var in 1:6
        old_values = [old_z[node_offset(i) + var] for i in 1:old_n]
        cursor = 1
        for (j, t) in enumerate(new_t)
            while cursor < old_n - 1 && old_t[cursor + 1] < t
                cursor += 1
            end
            value = if t <= old_t[1]
                old_values[1]
            elseif t >= old_t[end]
                old_values[end]
            else
                t0 = old_t[cursor]
                t1 = old_t[cursor + 1]
                w = (t - t0) / (t1 - t0)
                (1.0 - w) * old_values[cursor] + w * old_values[cursor + 1]
            end
            new_z[node_offset(j) + var] = value
        end
    end
    return new_z
end

function collocation_residual!(residual, z, p::ModelParams, steady::SteadyStateResult, tgrid::AbstractVector)
    N = length(tgrid)
    fill!(residual, 0.0)
    idx = 1

    y0 = node_slice(z, 1)
    residual[idx] = y0[1] - p.k0
    idx += 1
    residual[idx] = y0[3] - p.q0
    idx += 1
    residual[idx] = y0[5]
    idx += 1

    for i in 1:N-1
        yi = collect(node_slice(z, i))
        yj = collect(node_slice(z, i + 1))
        fi = dynamics(yi, p)
        fj = dynamics(yj, p)
        h = tgrid[i + 1] - tgrid[i]
        if !(all(isfinite, fi) && all(isfinite, fj))
            residual[idx:idx + 5] .= 1e6
            idx += 6
            continue
        end
        for j in 1:6
            residual[idx] = yj[j] - yi[j] - 0.5 * h * (fi[j] + fj[j])
            idx += 1
        end
    end

    yT = node_slice(z, N)
    residual[idx] = yT[1] - steady.k
    idx += 1
    residual[idx] = yT[2] - steady.c
    idx += 1
    residual[idx] = yT[3] - steady.q
end

function unpack_solution(z::AbstractVector, p::ModelParams, steady::SteadyStateResult, tgrid::AbstractVector, residual_norm::Real, success::Bool)
    N = length(tgrid)
    k = zeros(N)
    c = zeros(N)
    q = zeros(N)
    Λ1 = zeros(N)
    Λ2 = zeros(N)
    Λ3 = zeros(N)
    r_tilde = zeros(N)
    x = zeros(N)

    for i in 1:N
        yi = collect(node_slice(z, i))
        k[i], c[i], q[i], Λ1[i], Λ2[i], Λ3[i] = yi
        controls = foc_implied_controls(yi, p)
        if controls === nothing
            r_tilde[i] = NaN
            x[i] = NaN
        else
            r_tilde[i] = controls.r_tilde
            x[i] = controls.x
        end
    end

    return CollocationResult(success, collect(tgrid), k, c, q, Λ1, Λ2, Λ3, r_tilde, x, steady, Float64(residual_norm))
end

function solve_collocation(p::ModelParams = ModelParams(); N::Int = p.N, progress::Bool = true)
    steady = find_steady_state(p)
    stage_sizes = unique(max.(11, [cld(N, 3), cld(2 * N, 3), N]))

    previous_t = nothing
    previous_z = nothing
    final_result = nothing

    for Ncur in stage_sizes
        tgrid = range(0.0, p.T, length = Ncur)
        guess = previous_z === nothing ? collocation_guess(p, steady, tgrid) : interpolate_guess(previous_t, previous_z, tgrid)
        residual!(F, z) = collocation_residual!(F, z, p, steady, tgrid)
        progress && println("OptimalWealthTax collocation stage N=$(Ncur)")
        nls = nlsolve(residual!, guess; method = :trust_region, iterations = p.max_iter, xtol = 1e-10, ftol = 1e-10, show_trace = false)
        F = zeros(length(guess))
        residual!(F, nls.zero)
        resnorm = maximum(abs.(F))
        success = (nls.f_converged || nls.x_converged) && isfinite(resnorm) && resnorm <= p.residual_tolerance
        previous_t = collect(tgrid)
        previous_z = copy(nls.zero)
        final_result = unpack_solution(previous_z, p, steady, previous_t, resnorm, success)
    end

    return final_result
end