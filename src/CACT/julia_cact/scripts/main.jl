using DrWatson
@quickactivate("julia_cact")
using Revise

include(scriptsdir("run.jl"))

# using the DrWatson dictlist approacha: everything in a Vector is expanded once (Vectors of length 1 are not expanded naturally). 
allparams = Dict(
    "seed"=>collect(1:3),
    "true_base_location_generator"=>UniformLocationGenerator(),
    "auction_pool_location_generator"=>UniformLocationGenerator(),
    "x_min"=>0.0,
    "x_max"=>100.0,
    "y_min"=>0.0,
    "y_max"=>100.0,
    "size_auction_pool"=>12,
    "num_bundles"=>32,
    "true_num_locations"=>4,
    "pred_num_locations"=>4,
    "tsp_solver"=>ExactJuMPSolver(),
    "optimizer"=>NLOPT(max_eval=5),
    "x0_seeder"=>UniformRandomSeeder(),
    "proxy_objective_function"=>RMSE(),
    "true_objective_functions"=>Tuple(TrueObjectiveFunction[HausdorffDistance()]),
    "tags"=>Tuple(String["test",]),
)
dicts = dict_list(allparams)

for (i, d) in enumerate(dicts)
    config = CactConfig(; dict2ntuple(d))
    run_res = run(d)
    file_name = savename(d, "jld2")
    file_path = datadir("exp_raw", file_name)
    file_content = struct2dict(run_res)
    @tagsave(file_path, file_content)
end