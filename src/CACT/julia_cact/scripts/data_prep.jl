using DrWatson
using DataFrames
using JLD2
include("structs.jl")

function load_key(path::String, key)
    jldopen(path, "r") do file
        return file[key]
    end
end

function unpack_dict_column!(df::DataFrame, col::Symbol; prefix::String=string(col) * "__")
    # collect the union of keys across all rows, in case they differ
    all_keys = union((collect(keys(d)) for d in df[!, col])...)

    for k in all_keys
        new_col = Symbol(prefix, k)
        df[!, new_col] = [get(d, k, missing) for d in df[!, col]]
    end

    select!(df, Not(col))  # drop the original dict column
    return df
end


function combine_prefixed_columns!(df::DataFrame, prefix::String; new_col::Symbol=Symbol(prefix * "_combined"))
    prefix_with_sep = prefix * "__"
    matching_cols = filter(n -> startswith(n, prefix_with_sep), names(df))
    short_names = [replace(c, prefix_with_sep => "") for c in matching_cols]

    label_col = prefix in names(df) ? Symbol(prefix) : nothing

    df[!, new_col] = map(eachrow(df)) do row
        # skip fields the row's concrete type doesn't have (e.g. NoOpt has no algorithm/max_eval)
        parts = ["$(sn)=$(row[c])" for (sn, c) in zip(short_names, matching_cols) if !ismissing(row[c])]
        label = label_col === nothing ? titlecase(prefix) : string(row[label_col])
        isempty(parts) ? label : "$label(" * join(parts, ", ") * ")"
    end

    return df
end

function load_results(paths, key="optimize_result")
    results = [load_key(p, key) for p in paths]   # from before; swap to threaded version if needed

    multi_step_fields = [
        :x_trajectory,
        :incumbent_x_trajectory,
        :proxy_objective_trajectory,
        :incumbent_proxy_objective_trajectory,
        :true_objectives_trajectory,
        :incumbent_true_objectives_trajectory,
    ]
    multi_step_cols = Dict{Symbol,Any}(f => [getfield(r, f) for r in results] for f in multi_step_fields)
    # multi-step dataframe  
    msdf = DataFrame(; path=paths, multi_step_cols...)
    # add the steps
    msdf[!, :step] = [1:length(v) for v in msdf.proxy_objective_trajectory]
    # expand the true_objective dicts
    unpack_dict_column!(msdf, :true_objectives_trajectory)
    unpack_dict_column!(msdf, :incumbent_true_objectives_trajectory)
    trajectory_cols = filter(n -> n != :path && eltype(msdf[!, n]) <: Union{Missing,AbstractVector}, names(msdf, All()) .|> Symbol)
    msdf_long = DataFrames.flatten(msdf, trajectory_cols)
    # now remove all "_trajectory" parts from the column names
    rename!(msdf_long, names(msdf_long) .=> replace.(names(msdf_long), "_trajectory" => ""))

    single_step_fields = [
        :return_code,
        :num_evals,
        :x_opt,
        :proxy_objective_opt,
        :true_objectives_opt,
    ]
    single_step_cols = Dict{Symbol,Any}(f => [getfield(r, f) for r in results] for f in single_step_fields)
    # single step data frame
    ssdf = DataFrame(; path=paths, single_step_cols...)
    unpack_dict_column!(ssdf, :true_objectives_opt)

    return msdf_long, ssdf
end

"""
    add_summary_columns(gdf::GroupedDataFrame, cols::Vector{String})

For each column name in `cols`, computes mean and std within each group of `gdf`,
and adds `lower`/`upper` = mean ∓ std columns. Returns the combined DataFrame.
"""
function add_summary_columns(gdf::GroupedDataFrame, cols::Vector{String})
    pairs = [Symbol(c) => f => Symbol(c, "_", fname)
             for c in cols
             for (f, fname) in ((mean, "mean"), (std, "std"))]

    agg_df = combine(gdf, pairs...)

    for c in cols
        agg_df[!, Symbol(c, "_lower")] = agg_df[!, Symbol(c, "_mean")] .- agg_df[!, Symbol(c, "_std")]
        agg_df[!, Symbol(c, "_upper")] = agg_df[!, Symbol(c, "_mean")] .+ agg_df[!, Symbol(c, "_std")]
    end

    return agg_df
end

# ============================
function get_results(dir = datadir("exp_raw"))
    params_df = collect_results(dir, black_list=["input_data", "optimize_result", "config"])
    combine_prefixed_columns!(params_df, "optimizer")
    combine_prefixed_columns!(params_df, "tsp_solver")
    # grouped_params_df = groupby(params_df, Not([:path, :seed]))
    # here is where we first filter for the exact runs that we want, using e.g. the tag

    msdf, ssdf = load_results(params_df.path)
    # join with parameters
    msdf = outerjoin(params_df, msdf; on=:path)
    ssdf = outerjoin(params_df, ssdf; on=:path)

    # start with single step plots: boxplot per group

    # then use the multistep df to show the convergence behavior
    # group by some grouping vars; think: aggregate over different runs/seeds
    group_cols = [
        :proxy_objective_function,
        :optimizer_combined,
        :optimizer,
        :optimizer__algorithm,
        :optimizer__max_eval,
        :x0_seeder,
        :size_auction_pool,
        :true_num_locations,
        :pred_num_locations,
        # or all but path and seed?: [Symbol(n) for n in names(params_df) if !(n in (:path, :seed))]..., # this did not work as expected, back to hardcoded; ahh i forgot the value columns
        :step  # crucial to get mean/std/... per step
    ]
    grouped_msdf = groupby(msdf, group_cols)

    # create mean, std, mean-std and mean+std columns per group
    true_obj_cols = filter(n -> startswith(n, "true_objectives"), names(msdf))
    cols_to_summarize = ["proxy_objective"; "incumbent_proxy_objective"; true_obj_cols; ["incumbent_" * c for c in true_obj_cols]...]
    agg_msdf = add_summary_columns(grouped_msdf, cols_to_summarize)

    return params_df, ssdf, agg_msdf
end