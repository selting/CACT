using NLopt
using JLD2, UUIDs
using DrWatson
include("tsp.jl")
include("metrics.jl")
include("data_gen.jl")
include("plotting.jl")

abstract type DerivativeFreeOptimizer end

struct NLOPT <: DerivativeFreeOptimizer end

function auctioneer_optimize(;
    bundles,
    bids,
    pred_num_locations,
    _true_base_locations,
    opt_algorithm,
    params_lower_bounds,
    params_upper_bounds,
    x0,
    max_eval,
    proxy_objective_function::Symbol,
    true_objective_functions::Tuple{Vararg{Symbol}}
)
    num_parameters = 2 * pred_num_locations
    optimizer = NLopt.Opt(opt_algorithm, num_parameters)
    trajectory_buffer = []
    proxy_objective_buffer = []
    TrueObjectiveNT = NamedTuple{true_objective_functions}
    true_objective_buffer = TrueObjectiveNT[]

    # create the closure of the target_function that NLopt can handle
    partial_target_func = (x, grad) -> target_function(
        x=x,
        grad=grad,
        bundles=bundles,
        bids=bids,
        proxy_objective_func_symbol=proxy_objective_function,
        _true_base_locations=_true_base_locations,
        true_objective_funcs_symbols=true_objective_functions,
        trajectory_buffer=trajectory_buffer,
        proxy_objective_buffer=proxy_objective_buffer,
        true_objective_buffer=true_objective_buffer
    )
    NLopt.min_objective!(optimizer, partial_target_func)
    lower_bounds!(optimizer, repeat(params_lower_bounds, pred_num_locations))
    upper_bounds!(optimizer, repeat(params_upper_bounds, pred_num_locations))
    maxeval!(optimizer, max_eval)
    println("===== OPTIMIZE START ===== (x0: $x0)")
    opt_val, min_x, return_code = NLopt.optimize!(optimizer, x0)
    num_evals = NLopt.numevals(optimizer)
    x_opt = reshape(min_x, 2, :)'  # transpose because Julia is column major

    # TODO this should ideally check for success first, using variable ret (i.e. the return code)
    return Dict(
        "x_opt" => x_opt,
        "opt_val"=>opt_val,
        "num_evals" => num_evals,
        "return_code"=>return_code,
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
    # true_obj_fns = [TRUE_OBJECTIVE_REGISTRY[obj_fn] for obj_fn in true_objective_functions]
    # proxy_obj_fn = PROXY_OBJECTIVE_REGISTRY[proxy_objective_function]

    derived = generate_input_data(
        ; seed=seed,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        true_num_locations=true_num_locations,
        size_auction_pool=size_auction_pool,
        num_bundles=num_bundles,
        pred_num_locations=pred_num_locations
    )
    outputs = auctioneer_optimize(
        bundles=derived["bundles"],
        bids=derived["true_carrier_bids"],
        pred_num_locations=pred_num_locations,
        _true_base_locations=derived["true_base_locations"],
        opt_algorithm=opt_algorithm,
        params_lower_bounds=[x_min, y_min],
        params_upper_bounds=[x_max, y_max],
        x0=derived["x0"],
        max_eval=max_eval,
        proxy_objective_function=proxy_objective_function,
        true_objective_functions=true_objective_functions,
    )

    params = Dict{String,Any}(
        "seed" => seed,
        "x_min" => x_min,
        "x_max" => x_max,
        "y_min"=>y_min,
        "y_max"=>y_max,
        "true_num_locations"=>true_num_locations,
        "size_auction_pool"=>size_auction_pool,
        "num_bundles"=>num_bundles,
        "pred_num_locations" => pred_num_locations,
        "opt_algorithm" => opt_algorithm,
        "max_eval" => max_eval,
        "proxy_objective_function" => proxy_objective_function,
        "true_objective_functions" => true_objective_functions,
    )

    return Dict("params" => params, "derived" => derived, "outputs" => outputs)
end

function save_experiment(params::Dict, derived::Dict, output_dict::Dict;
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

experiment = run_experiment(
    seed=43,
    x_min=0,
    x_max=100,
    y_min=0,
    y_max=100,
    size_auction_pool=12,
    num_bundles=32,
    true_num_locations=4,
    pred_num_locations=4,
    # opt_algorithm=:GN_DIRECT_L_RAND,
    opt_algorithm=:GN_CRS2_LM,
    max_eval=10,
    proxy_objective_function=:RMSE,
    true_objective_functions=(:hausdorff_distance,)
)
tags = ["test_tag_v0.1"]
meta = Dict("version" => "v0.1")
plot_experiment(experiment["params"], experiment["derived"], experiment["outputs"])
save_experiment(experiment["params"], experiment["derived"], experiment["outputs"];  tags=tags, meta=meta)