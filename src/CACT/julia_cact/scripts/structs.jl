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
    x_min::Real =0
    x_max::Real =100
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
    true_objective_functions::Tuple{TrueObjectiveFunction}
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
    proxy_objective_trajectory::Vector{Float64}
    true_objectives_trajectory::Dict{Symbol, Vector{Float64}}
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
    true_objective_functions::Tuple{TrueObjectiveFunction}

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