# Visualization functions for Optimal Redistributive Capital Taxation Model

using PyPlot
using Statistics
using DelimitedFiles

"""
    plot_main_solution(result, title, filename; force=false)

Create the main solution plot with vertical layout (8 plots) matching the uploaded image layout.
Only saves to PNG, does not display. Set `force=true` to plot even if `result.success` is false.
"""
function plot_main_solution(result, title, filename; force::Bool=false)
    if !result.success && !force
        println("Cannot visualize - solution failed to converge (set force=true to override)")
        return
    end
    
    # Only plot data up to T/2 (first half of the solution)
    T_max = maximum(result.t)
    T_half = T_max / 2
    half_indices = result.t .<= T_half
    
    t_plot = result.t[half_indices]
    k_plot = result.k[half_indices]
    c_plot = result.c[half_indices]
    λ_plot = result.λ[half_indices]
    μ_plot = result.μ[half_indices]
    r_tilde_plot = result.r_tilde[half_indices]
    tau_k_plot = result.tau_k[half_indices]
    λ_transversality_plot = result.λ_transversality[half_indices]
    μ_transversality_plot = result.μ_transversality[half_indices]
    
    # Create 8 subplots in vertical layout (8 rows, 1 column)
    fig, ax = PyPlot.subplots(8, 1, figsize=(12, 16))
    fig.suptitle(title * " (0 to T/2)", fontsize=18)
    
    # Capital Trajectory (c as State) - Blue line
    ax[1].plot(t_plot, k_plot, "b-", linewidth=2)
    ax[1].set_title("Capital Trajectory (c as State)")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Capital k(t)")
    ax[1].grid(true)
    
    # Consumption Trajectory (State Variable) - Purple/magenta line
    ax[2].plot(t_plot, c_plot, "m-", linewidth=2)
    ax[2].set_title("Consumption Trajectory (State Variable)")
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("Consumption c(t)")
    ax[2].grid(true)
    
    # Costate λ Trajectory - Red line
    ax[3].plot(t_plot, λ_plot, "r-", linewidth=2)
    ax[3].set_title("Costate λ Trajectory")
    ax[3].set_xlabel("Time")
    ax[3].set_ylabel("Costate λ(t)")
    ax[3].grid(true)
    
    # Costate μ Trajectory - Orange line
    ax[4].plot(t_plot, μ_plot, color="orange", linewidth=2)
    ax[4].set_title("Costate μ Trajectory")
    ax[4].set_xlabel("Time")
    ax[4].set_ylabel("Costate μ(t)")
    ax[4].grid(true)
    
    # Effective Interest Rate - Green line
    ax[5].plot(t_plot, r_tilde_plot, "g-", linewidth=2)
    ax[5].set_title("Effective Interest Rate")
    ax[5].set_xlabel("Time")
    ax[5].set_ylabel("Effective Interest Rate")
    ax[5].grid(true)
    
    # Capital Tax Rate - Cyan line
    ax[6].plot(t_plot, tau_k_plot, "c-", linewidth=2)
    ax[6].set_title("Capital Tax Rate")
    ax[6].set_xlabel("Time")
    ax[6].set_ylabel("Capital Tax Rate")
    ax[6].grid(true)
    
    # λ Transversality Condition - Magenta line
    ax[7].plot(t_plot, λ_transversality_plot, "m-", linewidth=2)
    ax[7].set_title("λ Transversality Condition")
    ax[7].set_xlabel("Time")
    ax[7].set_ylabel("λ Transversality")
    ax[7].grid(true)
    
    # μ Transversality Condition - Red line
    ax[8].plot(t_plot, μ_transversality_plot, "r-", linewidth=2)
    ax[8].set_title("μ Transversality Condition")
    ax[8].set_xlabel("Time")
    ax[8].set_ylabel("μ Transversality")
    ax[8].grid(true)
    
    PyPlot.tight_layout()
    PyPlot.savefig(filename, dpi=300, bbox_inches="tight")
    PyPlot.close(fig)  # Close the figure to prevent display
    println("✓ Plot saved as '$filename'")
end
