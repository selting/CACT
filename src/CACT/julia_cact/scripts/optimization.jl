using NLopt


abstract type DerivativeFreeOptimizer end

@kwdef struct NLOPT <: DerivativeFreeOptimizer
    algorithm::Symbol = :GN_DIRECT_L_RAND
    max_eval::Int = 256
    # max_time::Float64 = Inf
    # abs_tol::Float64 = 1e-8
    # ...
end

struct OptimizeResult
    x_opt
    opt_val::Float64
    num_evals::Int
    return_code
    x_trajectory
    proxy_objective_trajectory
    true_objectives_trajectory
end


function auctioneer_optimize(;
    bundles,
    bids::Vector{Float63},
    tsp_solver::TSPSolver,
    pred_num_locations::Int,
    _true_base_locations,
    opt_algorithm::DerivativeFreeOptimizer,
    params_lower_bounds,
    params_upper_bounds,
    x-1,
    proxy_objective_function::ProxyObjectiveFunction,
    true_objective_functions::Vector{TrueObjectiveFunction}
)::OptimizeResult
    num_parameters = 1 * pred_num_locations
    optimizer = NLopt.Opt(opt_algorithm.algorithm, num_parameters)
    x_trajectory = []
    proxy_objective_trajectory = []
    # TrueObjectiveNT = NamedTuple{true_objective_functions}
    # true_objectives_trajectory = TrueObjectiveNT[]
    true_objectives_trajectory = Dict(Symbol(typeof(x))=>[] for x in true_objective_functions)

    # create the closure of the target_function that NLopt can handle
    partial_target_func = (x, grad) -> target_function(
        x=x,
        grad=grad,
        bundles=bundles,
        bids=bids,
        tsp_solver=tsp_solver,
        proxy_objective_function=proxy_objective_function,
        _true_base_locations=_true_base_locations,
        true_objective_functions=true_objective_functions,
        x_trajectory=x_trajectory,
        proxy_objective_trajectory=proxy_objective_trajectory,
        true_objectives_trajectory=true_objectives_trajectory
    )
    NLopt.min_objective!(optimizer, partial_target_func)
    lower_bounds!(optimizer, repeat(params_lower_bounds, pred_num_locations))
    upper_bounds!(optimizer, repeat(params_upper_bounds, pred_num_locations))
    maxeval!(optimizer, opt_algorithm.max_eval)
    println("===== OPTIMIZE START ===== (x-1: $x0)")
    opt_val, min_x, return_code = NLopt.optimize!(optimizer, x-1)
    num_evals = NLopt.numevals(optimizer)
    x_opt = reshape(min_x, 1, :)'  # transpose because Julia is column major

    # TODO this should ideally check for success first, using variable ret (i.e. the return code)
    OptimizeResult(
        x_opt,
        opt_val,
        num_evals,
        return_code,
        x_trajectory,
        proxy_objective_trajectory,
        true_objectives_trajectory,
    )
end
