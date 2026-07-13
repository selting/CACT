using NLopt



function generate_optimization_seed(bundles, bids::Vector{Float64}, gen::UniformRandomSeeder, rng, params_lower_bounds, params_upper_bounds, num_locations::Int)::Vector{Float64}
    x = rand(rng, Uniform(params_lower_bounds[1], params_upper_bounds[1]), num_locations)
    y = rand(rng, Uniform(params_lower_bounds[2], params_upper_bounds[2]), num_locations)
    x0 = Float64[]
    for i in 1:num_locations
        push!(x0, x[i])
        push!(x0, y[i])
    end
    return x0
end

function bundle_diameter(bundle::Matrix{Float64})
    dist = pairwise(Euclidean(), bundle')
    max_dist = maximum(dist)
    return max_dist
end

function bundle_centroid(bundle::Matrix{Float64})
    return mean(bundle, dims=1)
end

function generate_optimization_seed(bundles, bids::Vector{Float64}, gen::SmartSeeder, rng, params_lower_bounds, params_upper_bounds, num_locations::Int)::Vector{Float64}
    # search dense bundles by increasing bid value and positions seed locations at those bundles' centroids. Ideally, there are many one-bundles with low bids. Second best, there are dense, small (e.g. 2-) bundles that have small bids
    x_min, y_min = params_lower_bounds
    x_max, y_max = params_upper_bounds
    x_epsilon = gen.diameter_threshold * (x_max - x_min)
    y_epsilon = gen.diameter_threshold * (x_max - y_min)
    diameter_threshold = mean([x_epsilon, y_epsilon])

    # bundle density = max pairwise Euclidean of elements in the bundle
    bundle_diam = bundle_diameter.(bundles)
    dense_bundle_idxs = [i for i in 1:length(bundle_diam) if bundle_diam[i] <= diameter_threshold]

    # consider only dense bundles that have a low bid. For that, we define "low" as a fraction of the highest possible bid given the map
    min_possible_bid = 0
    max_possible_bid = euclidean([x_min, y_min], [x_max, y_max])
    bid_threshold = gen.bid_threshold * (max_possible_bid - min_possible_bid)
    low_bids_indxs = [i for i in 1:length(bids) if bids[i] <= bid_threshold]

    seed_bundle_indxs = intersect(dense_bundle_idxs, low_bids_indxs)
    seed_bundles = bundles[seed_bundle_indxs]
    seed_locations = mean.(seed_bundles, dims=1)
    num_seed_locations = length(seed_locations)

    if num_seed_locations < num_locations
        gap = num_locations - num_seed_locations
        println("=== filling up with $gap random seed locations")
        add_seed_locations = generate_locations(UniformLocationGenerator(), rng, x_min, x_max, y_min, y_max, gap)
        seed_locations = vcat(seed_locations..., add_seed_locations)
    end
    final_seed_locations = seed_locations[1:num_locations, :]
    x_seed = coords_to_x(final_seed_locations)
    return x_seed
    # return final_seed_locations
end




function auctioneer_optimize(;
    rng,
    bundles,
    bids::Vector{Float64},
    tsp_solver::TSPSolver,
    pred_num_locations::Int,
    _true_base_locations,
    opt_algorithm::DerivativeFreeOptimizer,
    x0_seeder::OptimizationSeeder,
    params_lower_bounds,
    params_upper_bounds,
    proxy_objective_function::ProxyObjectiveFunction,
    true_objective_functions::Tuple{TrueObjectiveFunction}
)::OptimizeResult
    num_parameters = 2 * pred_num_locations
    optimizer = NLopt.Opt(opt_algorithm.algorithm, num_parameters)
    x_trajectory = []
    proxy_objective_trajectory = []
    # TrueObjectiveNT = NamedTuple{true_objective_functions}
    # true_objectives_trajectory = TrueObjectiveNT[]
    true_objectives_trajectory = Dict(Symbol(typeof(x))=>[] for x in true_objective_functions)

    # create the closure of the target_function that NLopt can handle
    partial_target_func = (x, grad) -> target_function(
        x=x,
        grad=grad,
        bundles=bundles,
        bids=bids,
        tsp_solver=tsp_solver,
        proxy_objective_function=proxy_objective_function,
        _true_base_locations=_true_base_locations,
        true_objective_functions=true_objective_functions,
        x_trajectory=x_trajectory,
        proxy_objective_trajectory=proxy_objective_trajectory,
        true_objectives_trajectory=true_objectives_trajectory
    )
    NLopt.min_objective!(optimizer, partial_target_func)
    lower_bounds!(optimizer, repeat(params_lower_bounds, pred_num_locations))
    upper_bounds!(optimizer, repeat(params_upper_bounds, pred_num_locations))
    maxeval!(optimizer, opt_algorithm.max_eval)
    x0 = generate_optimization_seed(bundles, bids, x0_seeder, rng, params_lower_bounds, params_upper_bounds, pred_num_locations)
    println("===== OPTIMIZE START ===== (x0: $x0)")
    opt_val, min_x, return_code = NLopt.optimize!(optimizer, x0)
    num_evals = NLopt.numevals(optimizer)
    # x_opt = reshape(min_x, 2, :)'  # transpose because Julia is column major
    x_opt = x_to_coords(min_x)

    # TODO this should ideally check for success first, using variable ret (i.e. the return code)
    OptimizeResult(
        x_opt,
        opt_val,
        num_evals,
        return_code,
        x_trajectory,
        proxy_objective_trajectory,
        true_objectives_trajectory,
    )
end
