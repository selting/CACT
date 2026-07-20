# using CairoMakie  # is this the right thing to use here?!
using StatsPlots
using DrWatson
using DataFrames
include("structs.jl")

df = collect_results(datadir("exp_raw"), black_list=["input_data"])

gdf = groupby(df, ["optimizer", "x0_seeder"])

opt_val_df = combine(gdf, :optimize_result => or ->[row.opt_val for row in or]) 

@df opt_val_df violin(string.(:optimizer), :optimize_result_function, side=:left)
@df opt_val_df dotplot!(string.(:optimizer), :optimize_result_function, side=:right)