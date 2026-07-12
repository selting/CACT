using DrWatson

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
    x0_seeder::OptimizationSeeder
    proxy_objective_function::ProxyObjectiveFunction
    true_objective_functions::Tuple{TrueObjectiveFunction}
    tags::Tuple{String} # should this be part of the config?!
end

DrWatson.default_prefix(c::CactConfig) = "cact_"  #*c.tags[1]
DrWatson.allaccess(::CactConfig) = (
    :num_bundles,
    :true_base_location_generator,
    :auction_pool_locations_generator,
    :size_auction_pool,
    :true_num_loations,
    :pred_num_locations,
    :tsp_solver,
    :optimizer,
    :x0_seeder,
    :proxy_objective_function
)