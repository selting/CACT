using Makie
using julia_cact



function plot_optimize_result(r::OptimizeResult)
    n = r.num_evals
    iterations = 1:n
    proxy = r.proxy_objective_trajectory
    true_dict = r.true_objectives_trajectory
    keys_list = sort(collect(keys(true_dict)); by=string)

    # --- incumbent computation ---
    # running (iteration, proxy_value) pair with the smallest proxy_value seen so far
    incumbent_pairs = accumulate(
        (acc, x) -> x[2] < acc[2] ? x : acc,
        collect(zip(iterations, proxy)),
    )
    incumbent_idx = first.(incumbent_pairs)   # iteration at which each incumbent occurred
    incumbent_proxy = last.(incumbent_pairs)    # running-min proxy value

    # --- figure setup ---
    fig = Figure()
    ax = Axis(fig[1, 1],
        xlabel="Iteration",
        ylabel="Objective value",
        title="Optimization trajectory",
    )

    menu = Menu(fig[0, 1], options=[string(k) => k for k in keys_list])

    selected_key = Observable(keys_list[1])
    on(menu.selection) do sel
        selected_key[] = sel
    end

    # --- proxy objective (static, doesn't depend on menu selection) ---
    scatter!(ax, iterations, proxy;
        color=:blue, markersize=6, label="Proxy objective")

    lines!(ax, iterations, incumbent_proxy;
        color=:blue, linewidth=2, label="Incumbent (proxy)")

    # --- true objective (reactive: recomputes when menu selection changes) ---
    true_vals = @lift true_dict[$selected_key]
    scatter!(ax, iterations, true_vals;
        color=:red, markersize=6, label="True objective")

    incumbent_true = @lift true_dict[$selected_key][incumbent_idx]
    lines!(ax, iterations, incumbent_true;
        color=:red, linewidth=2, label="Incumbent (true)")

    axislegend(ax)

    return fig
end



function plot_aggregated_trajectory(agg::AggregatedOptimizeResult)
    n_evals = length(agg.incumbent_proxy_objective.mean)
    iterations = 1:n_evals
    keys_list = sort(collect(keys(agg.incumbent_true_objectives_trajectory)); by=string)

    fig = Figure()

    # --- left axis: proxy objective ---
    ax_proxy = Axis(fig[1, 1],
        xlabel="Iteration",
        ylabel="Proxy objective value",
        title="Incumbent trajectories (mean ± std, n=$(agg.n))",
        ylabelcolor=:blue,
        yticklabelcolor=:blue,
    )

    # --- right axis: true objective, sharing the same x-range ---
    ax_true = Axis(fig[1, 1],
        ylabel="True objective value",
        yaxisposition=:right,
        ylabelcolor=:orange,
        yticklabelcolor=:orange,
        limits=(nothing, nothing, 0, nothing)
    )
    hidespines!(ax_true)
    hidexdecorations!(ax_true)
    linkxaxes!(ax_proxy, ax_true)

    menu = Menu(fig[0, 1], options=[string(k) => k for k in keys_list])

    selected_key = Observable(keys_list[1])
    on(menu.selection) do sel
        selected_key[] = sel
    end

    # --- incumbent proxy: static band + mean line, on left axis ---
    proxy = agg.incumbent_proxy_objective
    proxy_lower = proxy.mean .- proxy.std
    proxy_upper = proxy.mean .+ proxy.std

    band!(ax_proxy, iterations, proxy_lower, proxy_upper;
        color=(:blue, 0.2), label="Incumbent proxy ± std")
    lines!(ax_proxy, iterations, proxy.mean;
        color=:blue, linewidth=2, label="Incumbent proxy mean")

    # --- incumbent true objective: reactive band + mean line, on right axis ---
    true_summary = @lift agg.incumbent_true_objectives_trajectory[$selected_key]

    true_lower = @lift $true_summary.mean .- $true_summary.std
    true_upper = @lift $true_summary.mean .+ $true_summary.std
    true_mean = @lift $true_summary.mean

    band!(ax_true, iterations, true_lower, true_upper;
        color=(:orange, 0.2), label="Incumbent $selected_key ± std")
    lines!(ax_true, iterations, true_mean;
        color=:orange, linewidth=2, label="Incumbent $selected_key mean")

    # --- combined legend, pulling entries from both axes ---
    # axislegend(ax_proxy, [ax_proxy.scene.plots; ax_true.scene.plots],
    #     ["Incumbent proxy ± std", "Incumbent proxy mean", "Incumbent true ± std", "Incumbent true mean"],
    #     )
    legend_elements = [
        LineElement(color=:blue, linewidth=2),
        LineElement(color=:orange, linewidth=2),
    ]
    legend_labels = ["Incumbent proxy mean", "Incumbent true mean"]

    Legend(fig[2, 1], legend_elements, legend_labels; orientation=:horizontal)

    return fig
end

# function plot_boxplots(gdf::GroupedDataFrame)

# end