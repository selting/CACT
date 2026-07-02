using JLD2
using DrWatson
using UUIDs
include("tsp.jl")
include("metrics.jl")
include("data_gen.jl")
include("plotting.jl")

@quickactivate("julia_cact")

struct RunResult
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
    proxy_objective_function::ProxyObjectiveFunction
    true_objective_funtions::Vector{TrueObjectiveFunction}

    # --- derived data (depends on seed/hyperparams, expensive to recompute) ---
    true_base_locations
    auction_pool_locations
    bundles
    bids

    # --- outputs (what the experiment produced) ---
    optimize_results::OptimizeResult

    # --- metadata ---
    run_idj
    timestamp
    # git_commit::String          # reproducibility — what code produced this
    # runtime_seconds::Float64
    tags::Vector{String}        # e.g. ["baseline", "debug", "paper-fig3"]
end



function run(;
    seed::Int,
    true_base_location_generator::LocationGenerator,
    auction_pool_location_generator::LocationGenerator,
    x_min::Int,
    x_max::Int,
    y_min::Int,
    y_max::Int,
    size_auction_pool::Int,
    num_bundles::Int,
    true_num_locations::Int,
    pred_num_locations::Int,
    tsp_solver::TSPSolver,
    optimizer::DerivativeFreeOptimizer,
    proxy_objective_function::ProxyObjectiveFunction,
    true_objective_functions::Vector{TrueObjectiveFunction},
)

    derived = generate_input_data(
        ; seed=seed,
        true_location_generator=true_base_location_generator,
        auction_pool_location_generator=auction_pool_location_generator,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        true_num_locations=true_num_locations,
        size_auction_pool=size_auction_pool,
        num_bundles=num_bundles,
        tsp_solver=tsp_solver,
        pred_num_locations=pred_num_locations
    )
    optimize_result = auctioneer_optimize(
        bundles=derived.bundles,
        bids=derived.true_carrier_bids,
        tsp_solver=tsp_solver,
        pred_num_locations=pred_num_locations,
        _true_base_locations=derived.true_base_locations,
        opt_algorithm=optimizer,
        params_lower_bounds=[x_min, y_min],
        params_upper_bounds=[x_max, y_max],
        x0=derived.x0,  # TODO add a seeding struct to enable efficient seeding based only on bundles and bids
        proxy_objective_function=proxy_objective_function,
        true_objective_functions=true_objective_functions,
    )

    return RunResult(
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
        proxy_objective_function,
        true_objective_functions,
        derived.true_base_locations,
        derived.auction_pool_locations,
        derived.bundles,
        derived.true_carrier_bids,
        optimize_result,
        uuid4(),
        time(),
        ["tag1", "tag2"]
    )
end

function save_run(params::Dict, derived::Dict, output_dict::Dict;
    tags::AbstractVector{String}=[],
    meta::AbstractDict{String}=Dict{String}())
    run_id = string(uuid4())
    path = joinpath("src/CACT/julia_cact/results", "$(run_id).jld2")
    tmp = path * ".tmp"
    jldsave(tmp;
        params=params,     # canonical — defines the experiment, tiny
        derived=derived,   # cache — regenerable from params, can be large
        outputs=output_dict,
        tags=tags,
        meta=meta)
    mv(tmp, path)
    return run_id
end

###################################################################
# RUNNING AN EXPERIMENT
###################################################################
run_result = run(
    seed=43,
    true_base_location_generator=UniformGenerator(),
    auction_pool_location_generator=UniformGenerator(),
    x_min=0,
    x_max=100,
    y_min=0,
    y_max=100,
    size_auction_pool=12,
    num_bundles=32,
    true_num_locations=4,
    pred_num_locations=4,
    tsp_solver=ExactJuMPSolver(),
    optimizer=NLOPT(max_eval=256),
    proxy_objective_function=RMSE(),
    true_objective_functions=TrueObjectiveFunction[HausdorffDistance()]
)
plot_run_result(run_result)
# tags = ["test_tag_v0.1"]
# meta = Dict("version" => "v0.1")
# plot_experiment(experiment["params"], experiment["derived"], experiment["outputs"])
# save_run(experiment["params"], experiment["derived"], experiment["outputs"]; tags=tags, meta=meta)