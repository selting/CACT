module julia_cact

# External dependencies, declared once for the whole library. Individual source
# files below assume these are in scope and do not `using`/`include` on their own.
using Statistics
using Distances
using Distributions
using Random
using JuMP
using HiGHS
using NLopt
using DrWatson

# Source files, included in dependency order. Each file assumes every file above
# it is already loaded -- there are no per-file includes anymore.
include("metrics.jl")            # objective functions (Hausdorff, RMSE, ...)
include("structs.jl")            # core types (solvers, optimizers, config, results)
include("config.jl")             # DrWatson serialization config for CactConfig
include("tsp.jl")                # TSP solvers (Held-Karp, HiGHS, nearest-neighbor)
include("target_function.jl")    # bid computation + optimization objective
include("data_gen.jl")           # location/bundle/input-data generation
include("optimization.jl")       # auctioneer_optimize dispatches
include("run.jl")                # single-run orchestration
include("utils.jl")              # misc helpers (flatten_dict)

# --- public API ---
# abstract interfaces
export LocationGenerator, TSPSolver, DerivativeFreeOptimizer, OptimizationSeeder
export ProxyObjectiveFunction, TrueObjectiveFunction
# location generators
export UniformLocationGenerator, ClusteredLocationGenerator, GridGenerator
# TSP solvers
export ExactJuMPSolver, NearestNeighborSolver, HeldKarpSolver, TwoOptSolver, TSPResult
# derivative-free optimizers
export NLOPT, NoOpt, RandomSearch
# seeders
export UniformRandomSeeder, SmartSeeder
# objective functions
export HausdorffDistance, NormalizedHausdorffDistance, TestDistance
export RMSE, MSE, HuberLoss
# config / data / results
export CactConfig, InputData, OptimizeResult, RunResult
# functions
export run_simulation, auctioneer_optimize, solve_tsp, compute_bids
export generate_input_data, generate_locations, target_function
export compute_true_objective, compute_proxy_objective
export x_to_coords, coords_to_x, flatten_dict

end # module
