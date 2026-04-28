if get(ENV, "MPLBACKEND", "") == ""
    ENV["MPLBACKEND"] = "Agg"
end

using PyPlot
PyPlot.ioff()

"""
    plot_solution(result, title, filename; force=false, half=false)

Plots and saves the `OptimalWealthTax` trajectory using nine stacked panels.

Input arguments:
- `result::CollocationResult`: solution to visualize; every trajectory stored inside has length `N = length(result.t)`.
- `title::AbstractString`: overall figure title.
- `filename::AbstractString`: path of the PNG file to write.

Optional parameters:
- `force::Bool=false`: save the plot even if `result.success == false`.
- `half::Bool=false`: if true, only use the first half of the horizon.

Output:
- Returns `filename` when the plot is saved.
- Returns `nothing` if the solution failed and `force == false`.
"""
function plot_solution(result::CollocationResult, title::AbstractString, filename::AbstractString; force::Bool = false, half::Bool = false)
    if !result.success && !force
        println("Cannot visualize - solution failed to converge (set force=true to override)")
        return nothing
    end

    inds = if half
        T_max = maximum(result.t)
        result.t .<= T_max / 2
    else
        trues(length(result.t))
    end

    t_plot = result.t[inds]
    k_plot = result.k[inds]
    c_plot = result.c[inds]
    q_plot = result.q[inds]
    Λ1_plot = result.Λ1[inds]
    Λ2_plot = result.Λ2[inds]
    Λ3_plot = result.Λ3[inds]
    r_plot = result.r_tilde[inds]
    x_plot = result.x[inds]

    scales = [
        max(abs(result.steady.k), 1.0),
        max(abs(result.steady.c), 1.0),
        max(abs(result.steady.q), 1.0),
        max(abs(result.steady.Λ1), 1.0),
        max(abs(result.steady.Λ2), 1.0),
        max(abs(result.steady.Λ3), 1.0),
    ]
    steady_distance = sqrt.(
        ((k_plot .- result.steady.k) ./ scales[1]) .^ 2 .+
        ((c_plot .- result.steady.c) ./ scales[2]) .^ 2 .+
        ((q_plot .- result.steady.q) ./ scales[3]) .^ 2 .+
        ((Λ1_plot .- result.steady.Λ1) ./ scales[4]) .^ 2 .+
        ((Λ2_plot .- result.steady.Λ2) ./ scales[5]) .^ 2 .+
        ((Λ3_plot .- result.steady.Λ3) ./ scales[6]) .^ 2
    )

    fig, ax = PyPlot.subplots(9, 1, figsize = (12, 18))
    fig.suptitle(title, fontsize = 18, y = 0.98)

    ax[1].plot(t_plot, k_plot, "b-", linewidth = 2)
    ax[1].axhline(result.steady.k, color = "k", linestyle = "--", linewidth = 1)
    ax[1].set_title("Capital k")
    ax[1].set_xlabel("Time")
    ax[1].grid(true)

    ax[2].plot(t_plot, c_plot, "m-", linewidth = 2)
    ax[2].axhline(result.steady.c, color = "k", linestyle = "--", linewidth = 1)
    ax[2].set_title("Consumption c")
    ax[2].set_xlabel("Time")
    ax[2].grid(true)

    ax[3].plot(t_plot, q_plot, color = "teal", linewidth = 2)
    ax[3].axhline(result.steady.q, color = "k", linestyle = "--", linewidth = 1)
    ax[3].set_title("Auxiliary state q")
    ax[3].set_xlabel("Time")
    ax[3].grid(true)

    ax[4].plot(t_plot, Λ1_plot, "r-", linewidth = 2)
    ax[4].axhline(result.steady.Λ1, color = "k", linestyle = "--", linewidth = 1)
    ax[4].set_title("Costate Lambda1")
    ax[4].set_xlabel("Time")
    ax[4].grid(true)

    ax[5].plot(t_plot, Λ2_plot, color = "orange", linewidth = 2)
    ax[5].axhline(result.steady.Λ2, color = "k", linestyle = "--", linewidth = 1)
    ax[5].set_title("Costate Lambda2")
    ax[5].set_xlabel("Time")
    ax[5].grid(true)

    ax[6].plot(t_plot, Λ3_plot, color = "brown", linewidth = 2)
    ax[6].axhline(result.steady.Λ3, color = "k", linestyle = "--", linewidth = 1)
    ax[6].set_title("Costate Lambda3")
    ax[6].set_xlabel("Time")
    ax[6].grid(true)

    ax[7].plot(t_plot, r_plot, "g-", linewidth = 2)
    ax[7].axhline(result.steady.r_tilde, color = "k", linestyle = "--", linewidth = 1)
    ax[7].set_title("Effective return r_tilde")
    ax[7].set_xlabel("Time")
    ax[7].grid(true)

    ax[8].plot(t_plot, x_plot, "k-", linewidth = 2)
    ax[8].axhline(result.steady.x, color = "gray", linestyle = "--", linewidth = 1)
    ax[8].set_title("Redistributed resources x")
    ax[8].set_xlabel("Time")
    ax[8].grid(true)

    ax[9].plot(t_plot, steady_distance, color = "purple", linewidth = 2)
    ax[9].set_title("Normalized distance to steady state")
    ax[9].set_xlabel("Time")
    ax[9].grid(true)

    PyPlot.tight_layout(rect = (0, 0, 1, 0.97))
    PyPlot.savefig(filename, dpi = 300, bbox_inches = "tight")
    PyPlot.close(fig)
    println("✓ Plot saved as '$(filename)'")
    return filename
end