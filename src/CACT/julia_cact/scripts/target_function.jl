include("metrics.jl")
include("tsp.jl")

function compute_bids(base_locations, bundles)
    # compute tsp without bundle
    objective_without_bundle = solve_tsp(base_locations)
    bids = []
    # TODO paralellize this!! one task per bundle
    for bundle in bundles
        # compute tsp with bundle
        tsp_locations = [base_locations; bundle]
        objective_with_bundle = solve_tsp(tsp_locations)
        bid = objective_with_bundle - objective_without_bundle
        push!(bids, bid)
    end
    return bids
end


#TODO let's skip the function caching for now, we might add that later

function target_function(;
    x::Vector,
    grad::Vector,
    bundles,
    bids,
    proxy_objective_func_symbol::Symbol,
    _true_base_locations::Matrix,
    true_objective_funcs_symbols::Tuple{Vararg{Symbol}},
    trajectory_buffer,
    proxy_objective_buffer,
    true_objective_buffer
)
    true_obj_fns = [TRUE_OBJECTIVE_REGISTRY[obj_fn] for obj_fn in true_objective_funcs_symbols]
    proxy_obj_fn = PROXY_OBJECTIVE_REGISTRY[proxy_objective_func_symbol]

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
    proxy_objective_value = proxy_obj_fn(bids_pred, bids)
    push!(proxy_objective_buffer, proxy_objective_value)

    # logging true objective
    TrueObjectiveNT = eltype(true_objective_buffer)
    # println("TrueObjecitveNT: $TrueObjectiveNT")
    vals = Tuple(f(base_locations_sorted, _true_base_locations) for f in true_obj_fns)
    # println("vals: $vals")
    valsNT = TrueObjectiveNT(vals)
    # println("valsNT: $valsNT")
    push!(true_objective_buffer, valsNT)
    # println("true_objective_buffer", true_objective_buffer)
    # println("---")

    num_evals = length(true_objective_buffer)
    println("Probe $num_evals")

    return proxy_objective_value
end
