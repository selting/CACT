using DrWatson
@quickactivate("julia_cact")
using Revise

include(scriptsdir("structs.jl"))
include(scriptsdir("run.jl"))

# using the DrWatson dictlist approacha: everything in a Vector is expanded once (Vectors of length 1 are not expanded naturally). 
allparams = Dict(
    "seed"=>collect(1:10),
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
    "optimizer"=>[NLOPT(max_eval=64), NoOpt()],
    "x0_seeder"=>UniformRandomSeeder(),
    "proxy_objective_function"=>RMSE(),
    "true_objective_functions"=>Tuple(TrueObjectiveFunction[HausdorffDistance()]),
    "tags"=>Tuple(String["wildboar",]),
)
dicts = dict_list(allparams)

function manage_run(config::CactConfig)
    input_data, optimize_result = run(config)
    file_content = merge(
        struct2dict(config),  # have config at the top level, this is how collect_results() expects it to be
        Dict(
            # "config" => struct2dict(config),  # TODO maybe storing plain dicts is better, but then i'd have to reconstruct the structs again in evaluation which is a pain I guess
            # "config" => config,
            # "input_data"=>struct2dict(input_data),
            "input_data" => input_data,
            # "optimize_result"=>struct2dict(optimize_result)
            "optimize_result" => optimize_result
        ))
    return file_content
end

path = datadir("exp_raw")
for (i, config_dict) in enumerate(dicts)
    config = CactConfig(; dict2ntuple(config_dict)...)
    data, file = produce_or_load(manage_run, config, path; filename=hash, prefix="cact")
end