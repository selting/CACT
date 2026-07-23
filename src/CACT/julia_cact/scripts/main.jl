using DrWatson
@quickactivate("julia_cact")
using Revise

include(scriptsdir("structs.jl"))
include(scriptsdir("run.jl"))
include("utils.jl")

# using the DrWatson dictlist approacha: everything in a Vector is expanded once (Vectors of length 1 are not expanded naturally). 
allparams = Dict(
    "seed"=>collect(1:50),
    "true_base_location_generator"=>UniformLocationGenerator(),
    "auction_pool_location_generator"=>UniformLocationGenerator(),
    "x_min"=>0.0,
    "x_max"=>100.0,
    "y_min"=>0.0,
    "y_max"=>100.0,
    "size_auction_pool"=>12,
    "num_bundles"=>64,
    "true_num_locations"=>8,
    "pred_num_locations"=>8,
    "tsp_solver"=>ExactJuMPSolver(),
    "optimizer" => [
        NoOpt(),
        NLOPT(:GN_DIRECT_L_RAND, 64),
        RandomSearch(64),
    ],
    "x0_seeder"=>UniformRandomSeeder(),
    "proxy_objective_function"=>RMSE(),
    "true_objective_functions"=>Tuple([
        HausdorffDistance(),
        NormalizedHausdorffDistance(0, 100, 0, 100)
    ]),
    "tags"=>Tuple(String["wildboar",]),
)
dicts = dict_list(allparams)

function manage_run(config_dict::Dict)
    display(config_dict)
    config = CactConfig(; dict2ntuple(config_dict)...)
    flat_config_dict = flatten_dict(config_dict)  

    # actual run
    input_data, optimize_result = run(config)

    # create the file to store
    # meta = Dict("date" => )
    file_content = merge(
        flat_config_dict,  # have config at the top level, this is how collect_results() expects it to be
        Dict(
            "config" => config,
            "input_data" => input_data,
            "optimize_result" => optimize_result
        ))
    return file_content
end

path = datadir("exp_raw")

results = Vector{Any}(undef, length(dicts))

Threads.@threads for i in eachindex(dicts)
    config_dict = dicts[i]
    data, file = produce_or_load(manage_run, config_dict, path; filename=hash, prefix="cact")
    results[i] = (data, file)
end