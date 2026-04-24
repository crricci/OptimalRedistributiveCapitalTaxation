if get(ENV, "MPLBACKEND", "") == ""
    ENV["MPLBACKEND"] = "Agg"
end

using PyPlot
PyPlot.ioff()

function plot_solution(result::CollocationResult, title::AbstractString, filename::AbstractString; force::Bool = false)
    if !result.success && !force
        println("Cannot visualize - solution failed to converge (set force=true to override)")
        return nothing
    end

    fig, ax = PyPlot.subplots(8, 1, figsize = (12, 18))
    fig.suptitle(title, fontsize = 18, y = 0.98)

    ax[1].plot(result.t, result.k, "b-", linewidth = 2)
    ax[1].set_title("Capital k")
    ax[1].grid(true)

    ax[2].plot(result.t, result.c, "m-", linewidth = 2)
    ax[2].set_title("Consumption c")
    ax[2].grid(true)

    ax[3].plot(result.t, result.q, color = "teal", linewidth = 2)
    ax[3].set_title("Auxiliary state q")
    ax[3].grid(true)

    ax[4].plot(result.t, result.Λ1, "r-", linewidth = 2)
    ax[4].set_title("Costate Lambda1")
    ax[4].grid(true)

    ax[5].plot(result.t, result.Λ2, color = "orange", linewidth = 2)
    ax[5].set_title("Costate Lambda2")
    ax[5].grid(true)

    ax[6].plot(result.t, result.Λ3, color = "brown", linewidth = 2)
    ax[6].set_title("Costate Lambda3")
    ax[6].grid(true)

    ax[7].plot(result.t, result.r_tilde, "g-", linewidth = 2)
    ax[7].set_title("Effective return r_tilde")
    ax[7].grid(true)

    ax[8].plot(result.t, result.x, "k-", linewidth = 2)
    ax[8].set_title("Redistributed resources x")
    ax[8].set_xlabel("Time")
    ax[8].grid(true)

    PyPlot.tight_layout(rect = (0, 0, 1, 0.97))
    PyPlot.savefig(filename, dpi = 300, bbox_inches = "tight")
    PyPlot.close(fig)
    println("✓ Plot saved as '$(filename)'")
    return filename
end