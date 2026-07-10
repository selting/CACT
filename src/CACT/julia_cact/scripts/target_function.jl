include("metrics.jl")
include("tsp.jl")

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
    true_objective_functions::Vector{TrueObjectiveFunction},
    x_trajectory,
    proxy_objective_trajectory,
    true_objectives_trajectory
)
    # true_obj_fns = [TRUE_OBJECTIVE_REGISTRY[obj_fn] for obj_fn in true_objective_funcs_symbols]
    # proxy_obj_fn = PROXY_OBJECTIVE_REGISTRY[proxy_objective_func_symbol]

    base_locations = x_to_coords(x)
    # this sorting will allow function caching later
    x_cache = Tuple(sortslices(base_locations, dims=1)')  # have to re-transpose to get tuple of (x1, y1, x2, y2, ...)
    # if x_cache is in the cache, skip the call to compute_bids, and retrieve bids from the cache instead

    # get the bids
    base_locations_sorted = sortslices(base_locations, dims=1)  # gives [x1 y1; x2 y2; ...]
    bids_pred = compute_bids(tsp_solver, base_locations_sorted, bundles)

    # logging the trajectory
    push!(x_trajectory, copy(base_locations_sorted))
    # not sure if copy is necessary in Julia

    # loggin proxy objective
    proxy_objective_value = compute_proxy_objective(proxy_objective_function, bids_pred, bids)
    push!(proxy_objective_trajectory, proxy_objective_value)

    # logging true objective
    # TrueObjectiveNT = eltype(true_objectives_trajectory)
    # vals = Tuple(f(base_locations_sorted, _true_base_locations) for f in true_obj_fns)
    # valsNT = TrueObjectiveNT(vals)
    # push!(true_objectives_trajectory, valsNT)
    for true_objective_func in true_objective_functions
        val = compute_true_objective(true_objective_func, base_locations_sorted, _true_base_locations)
        push!(true_objectives_trajectory[Symbol(typeof(true_objective_func))], val)
    end

    num_evals = length(proxy_objective_trajectory)
    println("Probe $num_evals: $proxy_objective_value")

    return proxy_objective_value
end
