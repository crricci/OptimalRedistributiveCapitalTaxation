function _segment_integral(base::Real, decay::Real, dt::Real)
    if abs(decay) <= sqrt(eps(Float64))
        return Float64(base) * Float64(dt)
    end
    return Float64(base) * (1.0 - exp(-Float64(decay) * Float64(dt))) / Float64(decay)
end

"""
    follower_best_response(t, r_tilde, a0; ρ, β, tail_rate=last(r_tilde))

Data una traiettoria esogena `r_tilde(t)`, calcola la best response esplicita del follower:

    c(t) = c0 * exp((R(t) - ρ t) / β),
    R(t) = ∫_0^t r_tilde(s) ds,

dove `c0` è ricostruito dalla condizione di trasversalità.

Input arguments:
- `t::AbstractVector{<:Real}`: griglia temporale strettamente crescente con `t[1] = 0`.
- `r_tilde::AbstractVector{<:Real}`: valori di `r_tilde` sulla stessa griglia.
- `a0::Real`: asset iniziale.

Optional parameters:
- `ρ::Real`: tasso di sconto.
- `β::Real`: parametro di curvatura dell'utilità.
- `tail_rate=last(r_tilde)`: valore costante assunto da `r_tilde` per `t > t[end]`.

Output:
- Restituisce un named tuple con scalari `c0`, `denominator`, `tail_rate`, `tail_decay`.
- Restituisce i vettori `t`, `r_tilde`, `R`, `c`, `a`, `tvc`, tutti di lunghezza `length(t)`.

Notes:
- Su ogni intervallo `[t[i], t[i+1])` la traiettoria `r_tilde` è trattata come costante e pari a `r_tilde[i]`.
- Per chiudere l'integrale all'infinito, oltre l'ultimo nodo si usa il tail costante `r_tilde(t) = tail_rate`.
"""
function follower_best_response(t::AbstractVector{<:Real}, r_tilde::AbstractVector{<:Real}, a0::Real;
    ρ::Real,
    β::Real,
    tail_rate::Real = Float64(r_tilde[end]))
    length(t) == length(r_tilde) || throw(ArgumentError("t and r_tilde must have the same length"))
    isempty(t) && throw(ArgumentError("t and r_tilde must be non-empty"))
    β > 0.0 || throw(ArgumentError("β must be strictly positive"))

    t_values = Float64.(t)
    r_values = Float64.(r_tilde)
    abs(t_values[1]) <= sqrt(eps(Float64)) || throw(ArgumentError("t[1] must be 0.0"))
    for i in 2:length(t_values)
        t_values[i] > t_values[i - 1] || throw(ArgumentError("t must be strictly increasing"))
    end

    n = length(t_values)
    R = zeros(Float64, n)
    kernel = zeros(Float64, n)
    spent_assets = zeros(Float64, n)
    denominator = 0.0

    kernel[1] = 1.0
    for i in 1:n-1
        dt = t_values[i + 1] - t_values[i]
        segment_rate = r_values[i]
        decay = (ρ + (β - 1.0) * segment_rate) / β
        base = exp(-(ρ * t_values[i] + (β - 1.0) * R[i]) / β)
        segment_budget = _segment_integral(base, decay, dt)
        denominator += segment_budget
        spent_assets[i + 1] = spent_assets[i] + segment_budget
        R[i + 1] = R[i] + segment_rate * dt
        kernel[i + 1] = exp(-(ρ * t_values[i + 1] + (β - 1.0) * R[i + 1]) / β)
    end

    tail_decay = Float64(ρ + (β - 1.0) * tail_rate)
    tail_decay > 0.0 || throw(ArgumentError("ρ + (β - 1) * tail_rate must be positive for the infinite-horizon tail integral to converge"))
    tail_weight = kernel[end]
    denominator += tail_weight * β / tail_decay

    c0 = Float64(a0) / denominator
    c = @. c0 * exp((R - ρ * t_values) / β)
    spent_assets .*= c0
    a = @. exp(R) * (a0 - spent_assets)
    tvc = @. c^(-β) * a * exp(-ρ * t_values)

    return (; t = t_values, r_tilde = r_values, R, c, a, c0, denominator, tail_rate = Float64(tail_rate), tail_decay, tvc)
end