using Makie
include("structs.jl")

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
    incumbent_idx   = first.(incumbent_pairs)   # iteration at which each incumbent occurred
    incumbent_proxy = last.(incumbent_pairs)    # running-min proxy value

    # --- figure setup ---
    fig = Figure()
    ax = Axis(fig[1, 1],
        xlabel = "Iteration",
        ylabel = "Objective value",
        title  = "Optimization trajectory",
    )

    menu = Menu(fig[0, 1], options = [string(k) => k for k in keys_list])

    selected_key = Observable(keys_list[1])
    on(menu.selection) do sel
        selected_key[] = sel
    end

    # --- proxy objective (static, doesn't depend on menu selection) ---
    scatter!(ax, iterations, proxy;
        color = :blue, markersize = 6, label = "Proxy objective")

    lines!(ax, iterations, incumbent_proxy;
        color = :blue, linewidth = 2, label = "Incumbent (proxy)")

    # --- true objective (reactive: recomputes when menu selection changes) ---
    true_vals = @lift true_dict[$selected_key]
    scatter!(ax, iterations, true_vals;
        color = :red, markersize = 6, label = "True objective")

    incumbent_true = @lift true_dict[$selected_key][incumbent_idx]
    lines!(ax, iterations, incumbent_true;
        color = :red, linewidth = 2, label = "Incumbent (true)")

    axislegend(ax, position = :rt)

    return fig
end