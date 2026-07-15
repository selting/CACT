using DrWatson
@quickactivate("julia_cact")
using Revise

include(scriptsdir("structs.jl"))
include(scriptsdir("run.jl"))

# using the DrWatson dictlist approacha: everything in a Vector is expanded once (Vectors of length 1 are not expanded naturally). 
allparams = Dict(
    "seed"=>collect(1:1000),
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
    "optimizer"=> NoOpt(),
    "x0_seeder"=>UniformRandomSeeder(),
    "proxy_objective_function"=>RMSE(),
    "true_objective_functions"=>Tuple(TrueObjectiveFunction[HausdorffDistance()]),
    "tags"=>Tuple(String["wildboar",]),
)
dicts = dict_list(allparams)

function flatten_dict(d::AbstractDict; sep::String = "__")
    flat = Dict{String, Any}()
    for (k, v) in d
        _flatten_into!(flat, string(k), v, sep)
    end
    return flat
end

# Decide whether a value should be treated as a "leaf" (kept as-is)
# or expanded further (dict / struct).
function _is_leaf(v)
    return v isa Union{AbstractString, Number, Symbol, Bool, Char,
                        Nothing, Missing, AbstractArray, Tuple, Function} ||
           v isa Type
end

_is_empty_struct(v) = isstructtype(typeof(v)) && fieldcount(typeof(v)) == 0

function _flatten_into!(flat::Dict{String, Any}, prefix::String, v, sep::String)
    if v isa AbstractDict
        for (k2, v2) in v
            _flatten_into!(flat, prefix * sep * string(k2), v2, sep)
        end
    elseif !_is_leaf(v) && _is_empty_struct(v)
        flat[prefix] = string(typeof(v))
    elseif !_is_leaf(v) && isstructtype(typeof(v))
        d2 = struct2dict(v)
        for (k2, v2) in d2
            _flatten_into!(flat, prefix * sep * string(k2), v2, sep)
        end
    elseif v isa AbstractVector || v isa Tuple
        flat[prefix] = Tuple(string(v2) for v2 in v)
    else
        flat[prefix] = v
    end
end

function manage_run(config_dict::Dict)
    config = CactConfig(; dict2ntuple(config_dict)...)
    flat_config_dict = flatten_dict(config_dict)  # not working well
    input_data, optimize_result = run(config)


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
for (i, config_dict) in enumerate(dicts)
    data, file = produce_or_load(manage_run, config_dict, path; filename=hash, prefix="cact")
end