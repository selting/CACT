using DrWatson

DrWatson.default_prefix(c::CactConfig) = "cact_"  #*c.tags[1]
DrWatson.default_allowed(::CactConfig) = (Real, String, LocationGenerator, TSPSolver, DerivativeFreeOptimizer, OptimizationSeeder, ProxyObjectiveFunction, TrueObjectiveFunction)
DrWatson.allaccess(::CactConfig) = (
    :seed,
    :num_bundles,
    :true_base_location_generator,
    :auction_pool_location_generator,
    :size_auction_pool,
    :true_num_locations,
    :pred_num_locations,
    :tsp_solver,
    :optimizer,
    :x0_seeder,
    :proxy_objective_function
)