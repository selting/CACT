include("metrics.jl")

# ========================================================

abstract type LocationGenerator end

struct UniformLocationGenerator <: LocationGenerator end
Base.string(::UniformLocationGenerator) = "UniformLocationGeneration"

struct ClusteredLocationGenerator <: LocationGenerator
    num_clusters::Int
    cluster_std::Float64
end
Base.string(g::ClusteredLocationGenerator) = "ClusteredLocationGenerator(#$g.num_clusters, std$g.cluster_std)"

struct GridGenerator <: LocationGenerator end

# ========================================================

abstract type TSPSolver end

@kwdef struct ExactJuMPSolver <: TSPSolver
    optimizer=HiGHS.Optimizer  # e.g. HiGHS.Optimizer
    time_limit=Inf
    mip_rel_gap=1e-4
end

struct NearestNeighborSolver <: TSPSolver end

@kwdef struct TwoOptSolver <: TSPSolver
    num_restarts::Int
    max_iterations::Int
end

struct TSPResult
    objective::Float64
    tour::Union{Vector{Int},Nothing}  # Nothing if you don't bother reconstructing it
    solve_time::Float64
    optimal
end

# ========================================================

abstract type DerivativeFreeOptimizer end

@kwdef struct NLOPT <: DerivativeFreeOptimizer
    algorithm::Symbol = :GN_DIRECT_L_RAND
    max_eval::Int = 256
    # max_time::Float64 = Inf
    # abs_tol::Float64 = 1e-8
    # ...
end

# Benchmark: Simply choose ONE single random estimation for x without optimizing at all
struct NoOpt <: DerivativeFreeOptimizer end

# ========================================================

abstract type OptimizationSeeder end

struct UniformRandomSeeder <: OptimizationSeeder end

struct SmartSeeder <: OptimizationSeeder
    diameter_threshold::Float64
    bid_threshold::Real
end

# ========================================================

@kwdef struct CactConfig
    seed::Int
    true_base_location_generator::LocationGenerator
    auction_pool_location_generator::LocationGenerator
    x_min::Real = 0
    x_max::Real = 100
    y_min::Real = 0
    y_max::Real = 100
    size_auction_pool::Int
    num_bundles::Int
    true_num_locations::Int
    pred_num_locations::Int
    tsp_solver::TSPSolver
    optimizer::DerivativeFreeOptimizer
    # TODO add a batch size: only batch_size out of all available bundles will be evaluated in each optimizer iteration. (randomly selected, or using a more advanced selection mechanism (e.g.: select harder bundles more often))
    x0_seeder::OptimizationSeeder
    proxy_objective_function::ProxyObjectiveFunction
    true_objective_functions::Tuple{Vararg{TrueObjectiveFunction}}
    tags::Tuple{String} # should this be part of the config?!
end

struct InputData
    true_base_locations::Matrix{Float64}
    auction_pool_locations::Matrix{Float64}
    bundles
    true_carrier_bids::Vector{Float64}
end

struct OptimizeResult
    x_opt
    opt_val::Float64
    num_evals::Int
    return_code
    x_trajectory
    incumbent_x_trajectory
    proxy_objective_trajectory::Vector{Float64}
    incumbent_proxy_objective_trajectory::Vector{Float64}
    true_objectives_trajectory::Dict{Symbol,Vector{Float64}}
    incumbent_true_objectives_trajectory::Dict{Symbol, Vector{Float64}}
end

struct AggrOptimizeResults
    opt_val_mean::Float64
    opt_val_std::Float64

    proxy_objective_trajectory_mean::Vector{Float64}
    proxy_objective_trajectory_std::Vector{Float64}

    true_objectives_trajectory_mean::Dict{Symbol,Vector{Float64}}
    true_objectives_trajectory_std::Dict{Symbol,Vector{Float64}}

end

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
    true_objective_functions::Tuple{Vararg{TrueObjectiveFunction}}

    # --- input_data data (depends on seed/hyperparams, expensive to recompute) ---
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


using Statistics

"""
    incumbent_trajectories(r::OptimizeResult)

Return (incumbent_idx, incumbent_proxy, incumbent_true) for a single run:
- incumbent_idx: at each iteration, the index of the best (lowest) proxy value seen so far
- incumbent_proxy: the running-min proxy value itself
- incumbent_true: Dict of the true-objective value at that same incumbent index, per key
"""
function incumbent_trajectories(r::OptimizeResult)
    proxy = r.proxy_objective_trajectory
    incumbent_pairs = accumulate(
        (acc, x) -> x[2] < acc[2] ? x : acc,
        collect(zip(eachindex(proxy), proxy)),
    )
    incumbent_idx = first.(incumbent_pairs)
    incumbent_proxy = last.(incumbent_pairs)

    incumbent_true = Dict(
        k => v[incumbent_idx] for (k, v) in r.true_objectives_trajectory
    )

    return incumbent_idx, incumbent_proxy, incumbent_true
end

struct ScalarSummary
    n::Int
    min::Float64
    q25::Float64
    mean::Float64
    q75::Float64
    max::Float64
    std::Float64
end

function summarize_scalar(xs::AbstractVector{<:Real})
    return ScalarSummary(
        length(xs),
        minimum(xs),
        quantile(xs, 0.25),
        mean(xs),
        quantile(xs, 0.75),
        maximum(xs),
        std(xs)
    )
end

struct TrajectorySummary
    n::Int
    min::Vector{Float64}
    q25::Vector{Float64}
    mean::Vector{Float64}
    q75::Vector{Float64}
    max::Vector{Float64}
    std::Vector{Float64}
end

function summarize_trajectories(vecs::AbstractVector{<:AbstractVector{<:Real}})
    M = stack(vecs; dims=1)
    return TrajectorySummary(
        length(vecs),
        vec(minimum(M; dims=1)),
        vec(mapslices(col -> quantile(col, 0.25), M; dims=1)),
        vec(mean(M; dims=1)),
        vec(mapslices(col->quantile(col, 0.75), M; dims=1)),
        vec(maximum(M; dims=1)),
        vec(std(M; dims=1)),
    )
end

struct AggregatedOptimizeResult
    opt_val::ScalarSummary
    proxy_objective_trajectory::TrajectorySummary
    true_objectives_trajectory::Dict{Symbol,TrajectorySummary}
    incumbent_proxy_objective::TrajectorySummary
    incumbent_true_objectives_trajectory::Dict{Symbol,TrajectorySummary}
    n::Int
end

function aggregate_results(vec_opt_res::AbstractVector)
    valid = collect(skipmissing(vec_opt_res))
    isempty(valid) && error("No non-missing OptimizeResult values to aggregate")

    # --- raw (non-incumbent) summaries, as before ---
    opt_val_summary = summarize_scalar([r.opt_val for r in valid])
    proxy_summary = summarize_trajectories([r.proxy_objective_trajectory for r in valid])

    all_keys = mapreduce(r -> collect(keys(r.true_objectives_trajectory)), union, valid)
    true_obj_summaries = Dict{Symbol,TrajectorySummary}(
        k => summarize_trajectories([r.true_objectives_trajectory[k] for r in valid])
        for k in all_keys
    )

    # --- per-run incumbent trajectories, then aggregate those ---
    incumbents = [incumbent_trajectories(r) for r in valid]  # Vector of (idx, proxy_inc, true_inc)

    incumbent_proxy_summary = summarize_trajectories([inc[2] for inc in incumbents])

    incumbent_true_obj_summaries = Dict{Symbol,TrajectorySummary}(
        k => summarize_trajectories([inc[3][k] for inc in incumbents])
        for k in all_keys
    )

    return AggregatedOptimizeResult(
        opt_val_summary,
        proxy_summary,
        true_obj_summaries,
        incumbent_proxy_summary,
        incumbent_true_obj_summaries,
        length(valid),
    )
end


#TODO struct for xmin, xmax, ymin, ymax (e.g. map or boundaries)