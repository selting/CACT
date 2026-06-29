using StatsBase
using JuMP
using HiGHS
using Distances
using Plots
using NLopt
using Random
using Distributions
using DataFrames
using StatsPlots
using JLD2, UUIDs

function draw_bundles(; rng, num_bundles::Int, auction_pool::Matrix)
    # use a set of tuples to track uniqueness cleanly
    unique_bundles = Set()
    bundles = []  # TODO better preallocate the space and populate in the loop
    pool_size = size(auction_pool, 1)

    while length(bundles) < num_bundles
        # 1 & 2: Random size between 1 and size_auction_pool (inclusive)
        bundle_size = rand(rng, 1:pool_size)

        # draw items without replacement
        bundle_indices = sample(rng, 1:pool_size, bundle_size, replace=false)
        bundle_ind_sorted = Tuple(sort(bundle_indices))

        # bundle=auction_pool[bundle_indices, :]

        if !(bundle_ind_sorted in unique_bundles)
            push!(unique_bundles, bundle_ind_sorted)
            push!(bundles, auction_pool[bundle_indices, :])
        end
    end
    return bundles
end

function solve_tsp(locations::Matrix)
    # solve the traveling salesperson problem using JuMP
    # copied from the JuMP tutorial website

    function build_tsp_model(d, n, optimizer)
        model = Model(optimizer)
        set_silent(model)
        @variable(model, x[1:n, 1:n], Bin, Symmetric)
        @objective(model, Min, sum(d .* x)/2)
        @constraint(model, [i in 1:n], sum(x[i, :]) == 2)  # flow conservation?
        @constraint(model, [i in 1:n], x[i, i] == 0)  # no self-loops
        return model
    end

    function subtour(edges::Vector{Tuple{Int,Int}}, n)
        shortest_subtour, unvisited = collect(1:n), Set(collect(1:n))
        while !isempty(unvisited)
            this_cycle, neighbors = Int[], unvisited
            while !isempty(neighbors)
                current = pop!(neighbors)
                push!(this_cycle, current)
                if length(this_cycle) > 1
                    pop!(unvisited, current)
                end
                neighbors = [j for (i, j) in edges if i == current && j in unvisited]
            end
            if length(this_cycle) < length(shortest_subtour)
                shortest_subtour = this_cycle
            end
        end
        return shortest_subtour
    end

    function selected_edges(x::Matrix{Float64}, n)
        return Tuple{Int,Int}[(i, j) for i in 1:n, j in 1:n if x[i, j] > 0.5]
    end

    subtour(x::Matrix{Float64}) = subtour(selected_edges(x, size(x, 1)), size(x, 1))
    subtour(x::AbstractMatrix{VariableRef}) = subtour(value.(x))

    # using the Iterative model: whenever a new subtour elimination constraint is added, start from scratch

    distances = pairwise(Euclidean(), locations')
    n = size(distances, 1)
    optimizer = HiGHS.Optimizer
    iterative_model = build_tsp_model(distances, n, optimizer)
    JuMP.optimize!(iterative_model)
    assert_is_solved_and_feasible(iterative_model)
    time_iterated = solve_time(iterative_model)
    cycle = subtour(iterative_model[:x])

    while 1 < length(cycle) < n
        # println("Found cycle of length $(length(cycle))")
        S = [(i, j) for (i, j) in Iterators.product(cycle, cycle) if i < j]
        @constraint(iterative_model, sum(iterative_model[:x][i, j] for (i, j) in S) <= length(cycle) - 1)
        JuMP.optimize!(iterative_model)
        assert_is_solved_and_feasible(iterative_model)
        # global time_iterated += solve_time(iterative_model)
        time_iterated += solve_time(iterative_model)
        # global cycle = subtour(iterative_model[:x])
        cycle = subtour(iterative_model[:x])
    end

    # println("Objective Value: $(objective_value(iterative_model))")
    # println("Runtime: $(time_iterated)")

    function plot_tour(X, Y, x)
        plot = Plots.plot()
        for (i, j) in selected_edges(x, size(x, 1))
            Plots.plot!([X[i], X[j]], [Y[i], Y[j]]; legend=false)
        end
        return plot
    end

    # X, Y = locations[:, 1], locations[:, 2]
    # plot_tour(X, Y, value.(iterative_model[:x]))

    return objective_value(iterative_model)
end


function compute_bids(base_locations, bundles)
    # compute tsp without bundle
    objective_without_bundle = solve_tsp(base_locations)
    bids = []  # TODO pre-allocate, size is know ex ante
    for bundle in bundles
        # compute tsp with bundle
        tsp_locations = [base_locations; bundle]
        objective_with_bundle = solve_tsp(tsp_locations)
        bid = objective_with_bundle - objective_without_bundle
        push!(bids, bid)
    end
    return bids
end

function rmse(y_pred::Vector, y::Vector)
    rmse = sqrt(mean((y_pred - y) .^ 2))
    return rmse
end

#TODO let's skip the function caching for now, we should add that later

function target_function(;
    x::Vector,
    grad::Vector,
    bundles,
    bids,
    proxy_objective_func,
    _true_base_locations::Matrix,
    true_objective_funcs,
    trajectory_buffer,
    proxy_objective_buffer,
    true_objective_buffer
)
    # println("- Evaluating target function at $x")
    base_locations = reshape(x, 2, :)'  # transposed view (adjoint) with ' because Julia is column major
    # this sorting will allow function caching later
    x_cache = Tuple(sortslices(base_locations, dims=1)')  # have to re-transpose to get tuple of (x1, y1, x2, y2, ...)
    # if x_cache is in the cache, skip the call to compute_bids, and retrieve bids from the cache instead

    # get the bids
    base_locations_sorted = sortslices(base_locations, dims=1)  # gives [x1 y1; x2 y2; ...]
    bids_pred = compute_bids(base_locations_sorted, bundles)
    # println("\tBids: $bids_pred")

    # logging the trajectory
    push!(trajectory_buffer, copy(base_locations_sorted))
    # not sure if copy is necessary in Julia

    # loggin proxy objective
    proxy_objective_value = proxy_objective_func(bids_pred, bids)
    push!(proxy_objective_buffer, proxy_objective_value)

    # logging true objective
    TrueObjectiveNT = eltype(true_objective_buffer)
    vals = Tuple(f(base_locations_sorted, _true_base_locations) for f in true_objective_funcs)
    push!(true_objective_buffer, TrueObjectiveNT(vals))

    num_evals = length(true_objective_buffer)
    print("Probe $num_evals")

    return proxy_objective_value
end

function hausdorff_distance(set_a, set_b)
    function directed_hausdorff_distance(set_a, set_b)
        # directed (pompeiou)-hausdorff distance: max of distances between each point x in set_a and its nearest neighbor y in set_b
        pairwise_dist = pairwise(Euclidean(), set_a, set_b, dims=1)
        nearest_neighbor_dist = minimum(pairwise_dist, dims=2)
        max_ = maximum(nearest_neighbor_dist)
        return max_
    end

    hausdorff = max(directed_hausdorff_distance(set_a, set_b), directed_hausdorff_distance(set_b, set_a))
    return hausdorff
end


function test_distance(set_a, set_b)
    return 0.5
end

function auctioneer_optimize(;
    bundles,
    bids,
    num_locations_to_estimate,
    _true_base_locations,
    opt_algorithm,
    params_lower_bounds,
    params_upper_bounds,
    rng,
    max_eval,
    proxy_objective_function,
    true_objective_functions
)
    num_parameters = 2 * num_locations_to_estimate
    optimizer = NLopt.Opt(opt_algorithm, num_parameters)
    trajectory_buffer = []
    proxy_objective_buffer = []
    TrueObjectiveNT = NamedTuple{nameof.(true_objective_functions)}
    true_objective_buffer = TrueObjectiveNT[]

    # create the closure of the target_function that NLopt can handle
    partial_target_func = (x, grad) -> target_function(
        x=x,
        grad=grad,
        bundles=bundles,
        bids=bids,
        proxy_objective_func=proxy_objective_function,
        _true_base_locations=_true_base_locations,
        true_objective_funcs=true_objective_functions,
        trajectory_buffer=trajectory_buffer,
        proxy_objective_buffer=proxy_objective_buffer,
        true_objective_buffer=true_objective_buffer
    )
    NLopt.min_objective!(optimizer, partial_target_func)
    lower_bounds!(optimizer, repeat(params_lower_bounds, num_locations_to_estimate))
    upper_bounds!(optimizer, repeat(params_upper_bounds, num_locations_to_estimate))
    maxeval!(optimizer, max_eval)
    x0 = rand.(rng, Uniform.(repeat(params_lower_bounds, num_locations_to_estimate), repeat(params_upper_bounds, num_locations_to_estimate)))
    println("===== OPTIMIZE START ===== (x0: $x0)")
    opt_val, min_x, ret = NLopt.optimize!(optimizer, x0)
    num_evals = NLopt.numevals(optimizer)
    x_opt = reshape(min_x, 2, :)'  # transpose because Julia is column major

    # TODO this should ideally check for success first using variable ret (i.e. the return code)
    return Dict(
        "x_opt" => x_opt,
        "opt_val"=>opt_val,
        "num_evals" => num_evals,
        "return_code"=>ret,
        "trajectory" => trajectory_buffer,
        "proxy_objective" => proxy_objective_buffer,
        "true_objective" => true_objective_buffer,
    )
end


function run_experiment(;
    seed::Int,
    x_min::Int,
    x_max::Int,
    y_min::Int,
    y_max::Int,
    size_auction_pool::Int,
    num_bundles::Int,
    true_num_locations::Int,
    pred_num_locations::Int,
    opt_algorithm,
    max_eval::Int,
    proxy_objective_function,
    true_objective_functions,
)
    rng = MersenneTwister(seed)

    true_base_locations_x = rand.(rng, Uniform.(x_min, x_max), true_num_locations)
    true_base_locations_y = rand.(rng, Uniform.(y_min, y_max), true_num_locations)
    true_base_locations = [true_base_locations_x true_base_locations_y]

    auction_pool_x = rand.(rng, Uniform.(x_min, x_max), size_auction_pool)
    auction_pool_y = rand.(rng, Uniform.(y_min, y_max), size_auction_pool)
    auction_pool = [auction_pool_x auction_pool_y]

    bundles = draw_bundles(rng=rng, num_bundles=num_bundles, auction_pool=auction_pool)
    true_carrier_bids = compute_bids(true_base_locations, bundles)

    optimize_result = auctioneer_optimize(
        bundles=bundles,
        bids=true_carrier_bids,
        num_locations_to_estimate=pred_num_locations,
        _true_base_locations=true_base_locations,
        opt_algorithm=opt_algorithm,
        params_lower_bounds=[x_min, y_min],
        params_upper_bounds=[x_max, y_max],
        rng=rng,
        max_eval=max_eval,
        proxy_objective_function=proxy_objective_function,
        true_objective_functions=true_objective_functions
    )

    return Dict(
        "inputs" => Dict(
            # function inputs
            "seed" => seed,
            "x_min" => x_min,
            "x_max" => x_max,
            "y_min" => y_min,
            "y_max" => y_max,
            "size_auction_pool" => size_auction_pool,
            "num_bundles" => num_bundles,
            "true_num_locations" => true_num_locations,
            "pred_num_locations" => pred_num_locations,
            "opt_algorithm" => opt_algorithm,
            "max_eval" => max_eval,
            "proxy_objective_function" => nameof(proxy_objective_function),
            "true_objective_function" => nameof.(true_objective_functions),
            # generated inside this function 
            "true_base_locations" => true_base_locations,
            "auction_pool" => auction_pool,
            "bundles" => bundles,
            "true_carrier_bids" => true_carrier_bids,
        ),
        "outputs" => optimize_result
    )
end

experiment = run_experiment(
    seed=42,
    x_min=0,
    x_max=100,
    y_min=0,
    y_max=100,
    size_auction_pool=12,
    num_bundles=64,
    true_num_locations=4,
    pred_num_locations=4,
    # opt_algorithm=:GN_DIRECT_L_RAND,
    opt_algorithm=:GN_CRS2_LM,
    max_eval=1024,
    proxy_objective_function=rmse,
    true_objective_functions=(hausdorff_distance,)
)
params = experiment["parameters"]
metrics = experiment["metrics"]


function save_experiment(input_dict, output_dict, meta::Dict)
    run_id = string(uuid4())
    path = joinpath("results", "$(run_id).jld2")
    tmp = path * ".tmp"
    jldsave(tmp; inputs=input_dict, outputs=output_dict, meta=meta)
    mv(tmp, path)  # atomic-ish rename: a crash mid-write never leaves a corrupt "real" file
    return run_id
end

function plot_experiment(experiment)
    # plot instance and estimations
    params = experiment["parameters"]
    metrics = experiment["metrics"]
    true_base_locations = params["true_base_locations"]
    auction_pool = params["auction_pool"]
    pred_base_locations = metrics["x_opt"]
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
    proxy_obj = metrics["proxy_objective"]
    true_obj = DataFrame(metrics["true_objective"])
    rolling_mins = accumulate(zip(proxy_obj, eachindex(proxy_obj))) do a, b
        a[1] <= b[1] ? a : b
    end
    rolling_min_indx = last.(rolling_mins)
    rolling_min_proxy = first.(rolling_mins)
    # get the values of the true objectives corresponding to rolling min of proxy
    rolling_min_true = true_obj[rolling_min_indx, :]

    p2 = scatter(proxy_obj, label=string(params["proxy_objective_function"]), alpha=0.5, color=:blue, ylims=(0, nothing))
    plot!(rolling_min_proxy, label="Rolling min of " * string(params["proxy_objective_function"]), color=:blue)

    scatter!(Matrix(true_obj), label=names(true_obj), alpha=0.5, color=:red)
    plot!(Matrix(rolling_min_true), label=names(true_obj), color=:red)
    xlabel!("Number of evaluations")
    plot!(legend=:outerbottom, legendcolumns=2)

    plot(p1, p2, size=(1000, 600) )
    title!(string(params["opt_algorithm"]))
end

plot_experiment(experiment)
