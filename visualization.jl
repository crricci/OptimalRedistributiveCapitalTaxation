# Visualization functions for Optimal Redistributive Capital Taxation Model

# Force non-interactive backend so figures are not shown
if get(ENV, "MPLBACKEND", "") == ""
    ENV["MPLBACKEND"] = "Agg"
end

using PyPlot
PyPlot.ioff()  # ensure interactive display is off
using DelimitedFiles

"""
    plot_main_solution(result, title, filename; force=false)

Create the main solution plot with vertical layout (8 plots) matching the uploaded image layout.
Only saves to PNG, does not display. Set `force=true` to plot even if `result.success` is false.
"""
function plot_main_solution(result, title, filename; force::Bool=false, half::Bool=true)
    if !result.success && !force
        println("Cannot visualize - solution failed to converge (set force=true to override)")
        return
    end
    
    # Optionally restrict to first half of time horizon
    inds = if half
        T_max = maximum(result.t)
        result.t .<= T_max/2
    else
        trues(length(result.t))
    end
    t_plot = result.t[inds]
    k_plot = result.k[inds]
    c_plot = result.c[inds]
    λ_plot = result.λ[inds]
    μ_plot = result.μ[inds]
    r_tilde_plot = result.r_tilde[inds]
    tau_k_plot = result.tau_k[inds]
    λ_transversality_plot = result.λ_tr[inds]
    μ_transversality_plot = result.μ_tr[inds]
    c_transversality_plot = result.c_tr[inds]
    
    # Create 9 subplots in vertical layout (9 rows, 1 column)
    fig, ax = PyPlot.subplots(9, 1, figsize=(12, 18))
    fig.suptitle(title, fontsize=18, y=0.98)
    
    # Capital Trajectory (c as State) - Blue line
    ax[1].plot(t_plot, k_plot, "b-", linewidth=2)
    ax[1].set_title("Capital k Trajectory")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel(" ")
    ax[1].grid(true)
    
    # Consumption Trajectory (State Variable) - Purple/magenta line
    ax[2].plot(t_plot, c_plot, "m-", linewidth=2)
    ax[2].set_title("Consumption c Trajectory")
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel(" ")
    ax[2].grid(true)
    
    # Costate λ Trajectory - Red line
    ax[3].plot(t_plot, λ_plot, "r-", linewidth=2)
    ax[3].set_title("Costate λ Trajectory")
    ax[3].set_xlabel("Time")
    ax[3].set_ylabel(" ")
    ax[3].grid(true)
    
    # Costate μ Trajectory - Orange line
    ax[4].plot(t_plot, μ_plot, color="orange", linewidth=2)
    ax[4].set_title("Costate μ Trajectory")
    ax[4].set_xlabel("Time")
    ax[4].set_ylabel(" ")
    ax[4].grid(true)
    
    # Effective Interest Rate - Green line
    ax[5].plot(t_plot, r_tilde_plot, "g-", linewidth=2)
    ax[5].set_title("Effective Interest Rate r̃")
    ax[5].set_xlabel("Time")
    ax[5].set_ylabel(" ")
    ax[5].grid(true)
    
    # Capital Tax Rate - Cyan line
    ax[6].plot(t_plot, tau_k_plot, "c-", linewidth=2)
    ax[6].set_title("Capital Tax Rate τₖ")
    ax[6].set_xlabel("Time")
    ax[6].set_ylabel(" ")
    ax[6].grid(true)
    
    # λ Transversality Condition - Magenta line
    ax[7].plot(t_plot, λ_transversality_plot, "m-", linewidth=2)
    ax[7].set_title("λ Transversality Condition")
    ax[7].set_xlabel("Time")
    ax[7].set_ylabel(" ")
    ax[7].grid(true)
    
    # μ Transversality Condition - Red line
    ax[8].plot(t_plot, μ_transversality_plot, "r-", linewidth=2)
    ax[8].set_title("μ Transversality Condition")
    ax[8].set_xlabel("Time")
    ax[8].set_ylabel(" ")
    ax[8].grid(true)

    # c Transversality Condition - Black dashed line
    ax[9].plot(t_plot, c_transversality_plot, "k--", linewidth=2)
    ax[9].set_title("c Transversality Condition")
    ax[9].set_xlabel("Time")
    ax[9].set_ylabel(" ")
    ax[9].grid(true)
    
    # Leave space for suptitle to avoid overlap with first subplot title
    PyPlot.tight_layout(rect=(0, 0, 1, 0.96))
    # Save without showing
    PyPlot.savefig(filename, dpi=300, bbox_inches="tight")
    PyPlot.close(fig)
    println("✓ Plot saved as '$filename'")
end


"""
    plot_welfare_vs_gamma(csvfile="welfare_gamma_scan.csv", outfile="welfare_vs_gamma.png"; title="Welfare vs γ")

Read the CSV produced by `run_gamma_welfare_scan` and plot γ on the x-axis vs welfare on the y-axis.
Saves the figure to `outfile` and does not display it.
"""
function plot_welfare_vs_gamma(csvfile::AbstractString = "welfare_gamma_scan.csv",
                               outfile::AbstractString = "welfare_vs_gamma.png";
                               title::AbstractString = "Welfare vs γ")
    data = readdlm(csvfile, ',')
    nrows, ncols = size(data)
    start_row = 1
    if nrows >= 1 && ncols >= 2
        # If header present, skip it
        if !(isa(data[1,1], Number) && isa(data[1,2], Number))
            start_row = 2
        end
    end
    gammas = Float64[]
    welfare = Float64[]
    for i in start_row:nrows
        gi = try
            Float64(data[i, 1])
        catch
            try
                parse(Float64, String(data[i, 1]))
            catch
                continue
            end
        end
        wi = try
            Float64(data[i, 2])
        catch
            try
                parse(Float64, String(data[i, 2]))
            catch
                continue
            end
        end
        if isfinite(gi) && isfinite(wi)
            push!(gammas, gi)
            push!(welfare, wi)
        end
    end
    if length(gammas) < 2
        println("Not enough valid data points to plot (need ≥ 2)")
        return nothing
    end
    # Sort by gamma for a clean line
    perm = sortperm(gammas)
    g = gammas[perm]
    w = welfare[perm]

    # Square figure
    fig, ax = PyPlot.subplots(1, 1, figsize=(8, 8))
    fig.suptitle(title, fontsize=16, y=0.96)
    ax.plot(g, w, "bo-", linewidth=2, markersize=4)
    ax.set_xlabel("γ")
    ax.set_ylabel("Welfare")
    ax.grid(true)
    PyPlot.tight_layout(rect=(0, 0, 1, 0.94))
    PyPlot.savefig(outfile, dpi=300, bbox_inches="tight")
    PyPlot.close(fig)
    println("✓ Gamma–welfare plot saved as '$(outfile)'")
    return outfile
end
