function compute_bids(tsp_solver::TSPSolver, base_locations, bundles)::Vector{Float64}
    # compute tsp without bundle
    res_without_bundle = solve_tsp(tsp_solver, base_locations)
    objective_without_bundle = res_without_bundle.objective
    bids = []
    # TODO paralellize this!! one task per bundle
    for bundle in bundles
        # compute tsp with bundle
        tsp_locations = [base_locations; bundle]
        res_with_bunbdle = solve_tsp(tsp_solver, tsp_locations)
        objective_with_bundle = res_with_bunbdle.objective
        bid = objective_with_bundle - objective_without_bundle
        push!(bids, bid)
    end
    return bids
end


#TODO let's skip the function caching for now, we might add that later

function x_to_coords(x)
    return reshape(x, 2, :)' # use ' to transpose as Julia is column major
end

function coords_to_x(coords)::Vector{Float64}
    return vcat(coords'...)
end

function target_function(;
    x::Vector,
    grad::Vector,
    bundles,
    bids,
    tsp_solver::TSPSolver,
    proxy_objective_function::ProxyObjectiveFunction,
    _true_base_locations::Matrix,
    true_objective_functions::Tuple{Vararg{TrueObjectiveFunction}},
    x_trajectory::AbstractVector,
    incumbent_x_trajectory::AbstractVector,
    proxy_objective_trajectory::AbstractVector,
    incumbent_proxy_objective_trajectory::AbstractVector,
    true_objectives_trajectory::AbstractDict,
    incumbent_true_objectives_trajectory::AbstractDict,
)

    base_locations = x_to_coords(x)
    # this sorting will allow function caching later
    x_cache = Tuple(sortslices(base_locations, dims=1)')  # have to re-transpose to get tuple of (x1, y1, x2, y2, ...)
    # if x_cache is in the cache, skip the call to compute_bids, and retrieve bids from the cache instead

    # 1. get the bids
    x_base_locations_sorted = sortslices(base_locations, dims=1)  # gives [x1 y1; x2 y2; ...]
    bids_pred = compute_bids(tsp_solver, x_base_locations_sorted, bundles)
    proxy_objective_value = compute_proxy_objective(proxy_objective_function, bids_pred, bids)
    
    # get the true objectives
    true_objective_values = Dict()  
    for true_objective_func in true_objective_functions
        key = Symbol(typeof(true_objective_func))
        val = compute_true_objective(true_objective_func, x_base_locations_sorted, _true_base_locations)
        true_objective_values[key] = val
    end

    # 2. find old and new incumbent
    if length(proxy_objective_trajectory) == 0
        old_incumbent_proxy = Inf64
        new_incumbent_proxy = proxy_objective_value
    else
        old_incumbent_proxy = last(incumbent_proxy_objective_trajectory)
        new_incumbent_proxy = min(proxy_objective_value, old_incumbent_proxy)
    end

    if new_incumbent_proxy < old_incumbent_proxy || length(proxy_objective_trajectory) == 0
        # println("new incumbent; update all incumbents")
        # we did find a new incumbent proxy objective 
        new_incumbent_x = x_base_locations_sorted
        new_incumbent_true = true_objective_values  # dict of values
    else
        # println("no new incumbent")
        # no new incumbent, copy the old incumbents
        new_incumbent_x = last(incumbent_x_trajectory)
        new_incumbent_true = Dict()
        for true_objective_func in true_objective_functions
            key = Symbol(typeof(true_objective_func))
            val = last(incumbent_true_objectives_trajectory[key])
            new_incumbent_true[key] = val
        end
    end

    # 3. logging the trajectories
    push!(x_trajectory, copy(x_base_locations_sorted))  # x trajectory
    push!(proxy_objective_trajectory, proxy_objective_value)  # proxy trajectory
    # true trajectory:
    for (key, val) in true_objective_values
        push!(true_objectives_trajectory[key], val)
    end

    # 4. log the incumbents
    push!(incumbent_proxy_objective_trajectory, new_incumbent_proxy)
    push!(incumbent_x_trajectory, new_incumbent_x)
    for (key, val) in new_incumbent_true
        push!(incumbent_true_objectives_trajectory[key], val)
    end

    # num_evals = length(proxy_objective_trajectory)
    # println("Probe $num_evals: $proxy_objective_value")

    return proxy_objective_value
end
