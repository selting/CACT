using Plots
using StatsPlots
using DataFrames
function plot_experiment(params, derived, outputs)
    # plot instance and estimations
    true_base_locations = derived["true_base_locations"]
    auction_pool = derived["auction_pool"]
    pred_base_locations = outputs["x_opt"]
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
    proxy_obj = outputs["proxy_objective"]
    true_obj = DataFrame(outputs["true_objective"])
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
        label=string(params["proxy_objective_function"]),
        alpha=0.2,
        color=:blue,
        size=1,
        # ylims=(0, nothing)
    )
    plot!(
        rolling_min_proxy_obj,
        label="Rolling min of " * string(params["proxy_objective_function"]), color=:blue
    )

    # TRUE objective(s)
    scatter!(
        Matrix(true_obj),
        label=names(true_obj),
        alpha=0.5,
        color=:red
    )
    plot!(Matrix(rolling_true_obj), label=names(true_obj), color=:red)
    xlabel!("Number of evaluations")
    plot!(legend=:outerbottom, legendcolumns=2)

    plot(p1, p2, layout=(1, 2), size=(1000, 600))
    title!(string(params["opt_algorithm"]))
end

