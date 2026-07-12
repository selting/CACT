using JLD2
using UUIDs
include("tsp.jl")
include("metrics.jl")
include("data_gen.jl")
include("optimization.jl")
include("config.jl")


@kwdef struct RunResult
    # --- hyperparameters (the "what did we configure") ---
    seed::Int
    true_location_generator::LocationGenerator
    auction_pool_location_generator::LocationGenerator
    x_min::Int
    x_max::Int
    y_min::Int
    y_max::Int
    size_auction_pool::Int
    num_bundles::Int
    pred_num_locations::Int
    tsp_solver::TSPSolver
    optimizer::DerivativeFreeOptimizer
    x0_seeder::OptimizationSeeder
    proxy_objective_function::ProxyObjectiveFunction
    true_objective_functions::Tuple{TrueObjectiveFunction}

    # --- derived data (depends on seed/hyperparams, expensive to recompute) ---
    true_base_locations
    auction_pool_locations
    bundles
    bids

    # --- outputs (what the experiment produced) ---
    optimize_results::OptimizeResult

    # --- metadata ---
    run_id
    timestamp
    # git_commit::String          # reproducibility — what code produced this
    # runtime_seconds::Float64
    tags::Tuple{String}        # e.g. ["baseline", "debug", "paper-fig3"]
end


# function save_run(params::Dict, derived::Dict, output_dict::Dict;
#     tags::AbstractVector{String}=[],
#     meta::AbstractDict{String}=Dict{String}())
#     run_id = string(uuid4())
#     path = joinpath("src/CACT/julia_cact/results", "$(run_id).jld2")
#     tmp = path * ".tmp"
#     jldsave(tmp;
#         params=params,     # canonical — defines the experiment, tiny
#         derived=derived,   # cache — regenerable from params, can be large
#         outputs=output_dict,
#         tags=tags,
#         meta=meta)
#     mv(tmp, path)
#     return run_id
# end

# function save_run(res::RunResult)
#     runs_dir = datadir("exp_raw")
#     name = res.run_id
#     path = joinpath(runs_dir, "$name.jld2")
#     jldsave(path; res)
# end


function run(config::CactConfig)

    rng = Xoshiro(config.seed)

    derived = generate_input_data(;
        rng=rng,
        true_location_generator=config.true_base_location_generator,
        auction_pool_location_generator=config.auction_pool_location_generator,
        x_min=config.x_min,
        x_max=config.x_max,
        y_min=config.y_min,
        y_max=config.y_max,
        true_num_locations=config.true_num_locations,
        size_auction_pool=config.size_auction_pool,
        num_bundles=config.num_bundles,
        tsp_solver=config.tsp_solver,
        pred_num_locations=config.pred_num_locations
    )
    optimize_result = auctioneer_optimize(
        rng=rng,
        bundles=derived.bundles,
        bids=derived.true_carrier_bids,
        tsp_solver=tsp_solver,
        pred_num_locations=pred_num_locations,
        _true_base_locations=derived.true_base_locations,
        opt_algorithm=optimizer,
        x0_seeder=x0_seeder,
        params_lower_bounds=[x_min, y_min],  # TODO avoid the back and forth from x_min, x_max, y_min, y_max to vectors of min and max values. I guess its best to stick to xmin, xmax where possible
        params_upper_bounds=[x_max, y_max],
        proxy_objective_function=proxy_objective_function,
        true_objective_functions=true_objective_functions,
    )

    res = RunResult(
        seed,
        true_base_location_generator,
        auction_pool_location_generator,
        x_min,
        x_max,
        y_min,
        y_max,
        size_auction_pool,
        num_bundles,
        pred_num_locations,
        tsp_solver,
        optimizer,
        x0_seeder,
        proxy_objective_function,
        true_objective_functions,
        derived.true_base_locations,
        derived.auction_pool_locations,
        derived.bundles,
        derived.true_carrier_bids,
        optimize_result,
        uuid4(),
        time(),
        tags,
    )

    # save_run(res)
    return res
end
