using Plots
using StatsPlots
using DataFrames

function plot_run_result(res::RunResult)
    # plot instance and estimations
    true_base_locations = res.true_base_locations
    auction_pool = res.auction_pool_locations
    pred_base_locations = res.optimize_results.x_opt
    p1 = scatter(
        true_base_locations[:, 1],
        true_base_locations[:, 2],
        label="True base locations",
        shape=:square
    )
    scatter!(
        auction_pool[:, 1],
        auction_pool[:, 2],
        label="Auction pool",
        shape=:circle
    )
    scatter!(
        pred_base_locations[:, 1],
        pred_base_locations[:, 2],
        label="Predicted base locations",
        shape=:diamond
    )
    xlabel!("x")
    ylabel!("y")
    plot!(legend=:outerbottom, legendcolumns=3)

    # plot the learning curves
    proxy_obj = res.optimize_results.proxy_objective_trajectory
    true_obj = DataFrame(res.optimize_results.true_objectives_trajectory)
    rolling_mins = accumulate(zip(proxy_obj, eachindex(proxy_obj))) do a, b
        a[1] <= b[1] ? a : b
    end
    rolling_min_indx = last.(rolling_mins)
    rolling_min_proxy_obj = first.(rolling_mins)
    # get the values of the true objectives corresponding to rolling min of proxy
    rolling_true_obj = true_obj[rolling_min_indx, :]

    # PROXY objective
    p2 = scatter(
        proxy_obj,
        label=string(Symbol(res.proxy_objective_function)),
        alpha=0.2,
        color=:blue,
        markersize=3,
        # ylims=(0, nothing)
    )
    plot!(p2,
        rolling_min_proxy_obj,
        label="Rolling min of " * string(res.proxy_objective_function), color=:blue
    )

    # TRUE objective(s)
    for col in names(true_obj)
        scatter!(p2, true_obj[!, col], label=col, alpha=0.5, markersize=3, color=:red)
        plot!(p2, rolling_true_obj[!, col], label=col * "rolling min", color=:red)
    end

    xlabel!(p2, "Number of evaluations")
    plot!(p2, legend=:outerbottom, legendcolumns=2)

    plot(p1, p2, layout=(1, 2), size=(1000, 600))
    title!(string(res.optimizer))
end