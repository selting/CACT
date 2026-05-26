from functools import cache, partial

from joblib import Parallel, delayed
import nlopt
import numpy as np
from auction_module.bundling_and_bidding.fitness_functions.vrp_learn.distance import (
    my_convex_hull_jaccard_distance,
    my_hausdorff_distance,
    my_modified_hausdorff_distance,
)
from pyvrp import Model
from pyvrp.stop import MaxRuntime


def draw_bundles(
    rng: np.random.Generator, size_auction_pool, num_bundles, auction_pool: tuple[tuple]
):
    # Use a set of tuples to track uniqueness cleanly without broadcasting issues
    unique_bundles = set()
    bundles = []

    # Just in case: handle a pool array that might be multi-dimensional or a list
    # auction_pool = np.asarray(auction_pool)

    while len(bundles) < num_bundles:
        # 1 & 2. Random size between 1 and size_auction_pool (inclusive)
        bundle_size = rng.choice(np.arange(1, size_auction_pool + 1))

        # Draw items WITHOUT replacement so a single bundle doesn't have internal duplicates
        bundle_ind = rng.choice(len(auction_pool), size=bundle_size, replace=False)
        bundle_tuple = tuple(auction_pool[i] for i in bundle_ind)

        # Sort the bundle elements to ensure that bundles with identical items
        # in different orders (e.g., [1, 2] and [2, 1]) are flagged as duplicates
        bundle_sorted_tuple = tuple(sorted(bundle_tuple))

        # 3. Check for duplicates safely using the set
        if bundle_sorted_tuple not in unique_bundles:
            unique_bundles.add(bundle_sorted_tuple)
            # Append the actual numpy array to your final list
            bundles.append(bundle_tuple)

    return tuple(bundles)


def solve_tsp(locations):
    "solve a tsp with given locations (2d np.array) using PyVRP. Other solvers could be used, e.g. dynamic programming for small enough instances."
    m = Model()
    m.add_vehicle_type(1)
    depot = m.add_depot(x=locations[0][0], y=locations[0][1])
    clients = [
        m.add_client(x=locations[idx][0], y=locations[idx][1])
        for idx in range(1, len(locations))
    ]

    locations = [depot] + clients
    for frm in locations:
        for to in locations:
            distance = abs(frm.x - to.x) + abs(frm.y - to.y)  # Manhattan
            m.add_edge(frm, to, distance=distance)

    res = m.solve(stop=MaxRuntime(0.5), display=False)  # one second
    min_cost = res.best.distance_cost()
    return min_cost


# def solve_tsp_DP(locations:np.array):
#     dist = squareform(pdist(locations))
#     n = len(locations)
#
#     # memoization for top down recursion
#     memo = [[-1]*(1 << (n+1)) for _ in range(n+1)]
#
#     def fun(i, mask):
#         # base case
#         # if only ith bit and 1st bit is set in our mask,
#         # it implies we have visited all other nodes already
#         if mask == ((1 << i) | 3):
#             return dist[1][i]
#
#         # memoization
#         if memo[i][mask] != -1:
#             return memo[i][mask]
#
#         res = 10**9  # result of this sub-problem
#
#         # we have to travel all nodes j in mask and end the path at ith node
#         # so for every node j in mask, recursively calculate cost of
#         # travelling all nodes in mask
#         # except i and then travel back from node j to node i taking
#         # the shortest path take the minimum of all possible j nodes
#         for j in range(1, n+1):
#             if (mask & (1 << j)) != 0 and j != i and j != 1:
#                 res = min(res, fun(j, mask & (~(1 << i))) + dist[j][i])
#         memo[i][mask] = res  # storing the minimum value
#         return res
#
#
#     # Driver program to test above logic
#     ans = 10**9
#     for i in range(1, n+1):
#         # try to go from node 1 visiting all nodes in between to i
#         # then return from i taking the shortest route to 1
#         ans = min(ans, fun(i, (1 << (n+1))-1) + dist[i][1])
#
#     print("The cost of most efficient tour = " + str(ans))


def compute_bids(base_locations: np.array, bundles: list[np.array]):
    # compute tsp without bundle
    objective_without_bundle = solve_tsp(base_locations)
    bids = []
    for bundle in bundles:
        # compute tsp with bundle
        tsp_locations = np.concat([base_locations, bundle], axis=0)
        objecvtive_with_bundle = solve_tsp(tsp_locations)
        bid = objecvtive_with_bundle - objective_without_bundle
        bids.append(bid)
    return bids


def _worker(base_locations, bundle, objective_without_bundle, solver_func):
    tsp_locations = np.concatenate([base_locations, bundle], axis=0)
    # Using a passed solver function for future flexibility
    objective_with_bundle = solver_func(tsp_locations)
    return objective_with_bundle - objective_without_bundle

def compute_bids_parallel(
    base_locations: np.array, 
    bundles: list[np.array], 
    solver_func=solve_tsp, 
    n_jobs=-1
):
    # Baseline calculation
    objective_without_bundle = solver_func(base_locations)
    
    # CURRENTLY: Use "threading" because PyVRP is C++ based
    # FUTURE: If you switch to a pure Python solver, just change backend to "multiprocessing"
    bids = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_worker)(base_locations, bundle, objective_without_bundle, solver_func)
        for bundle in bundles
    )
    
    return bids


def rmse(y_pred, y):
    rmse = np.sqrt(np.mean((np.array(y_pred) - np.array(y)) ** 2))
    return rmse


@cache
def _evaluate_candidate_cached(
    x: tuple[float], bundles: tuple[tuple[tuple[float, float]]]
):
    """basically just a cached version of the compute_bids function to avoid costly recomputes if "the same" inputs have been given in the past already. "Same" in this application includes also permutations of previous candidates.

    Args:
        x (tuple[float]): the candidate parameters tested by a derivative-free black box optimizer, such as those of NLopt
        bundles (tuple[tuple[tuple[float, float]]]): the bundles that are queried

    Returns:
        _type_: the 2D coordinate representation of the candidate input x (just a reshape) AND the predicted bids, assuming those coordinates were the base locations.
    """
    # Reconstruct the correct 2d shape
    x_2d = np.array(x).reshape(-1, 2, copy=True)

    # Run the expensive bidding simulation
    # bids_pred = compute_bids(x_2d, bundles)
    # test: parallel tsp solving
    bids_pred = compute_bids_parallel(x_2d, bundles)
    return x_2d, bids_pred


def target_function(
    x: np.array,
    grad,
    bundles: tuple[tuple[tuple]],
    bids,
    proxy_objective_func,
    _true_base_locations: np.array,
    true_objective_funcs: list[callable],
    trajectory_buffer: list,
    proxy_objective_buffer: list,
    true_objective_buffer: list,
):
    """the user (and NLopt) -facing target function. Best to be used with functools.partial to obtain the version that
    accepts only x (and a gradient).

    Args:
        x (np.array): candidate parameters (base location coords) to evaluate as a 1D array [x1, y1, x2, y2, ...]
        grad (_type_): potentially a gradient, omitted in my case
        bundles (tuple[tuple[tuple]]): the bundles that have been queried
        bids (_type_): the bids that the carrier reported
        proxy_objective_func (_type_): the loss to minimize, usually rmse
        _true_base_locations (np.array): the actual base locations, leading underscore to highlight that these are not actually available during the optimization. Only used for tracking/logging.
        true_objective_funcs (list[callable]): a list of set difference functions such as the hausdorff distance. must take as input two 2d sequences.
        trajectory_buffer (list): and empty list that will be modified in-place to track the candidates that were evaluated
        proxy_objective_buffer (list): empty list to track obj function values
        true_objective_buffer (list): empty list to store true objective function values

    Returns:
        float: the obj value of the proxy obj function
    """
    # Normalize input by sorting in 2D shape. This is what allows proper function caching.
    pairs = x.reshape(-1, 2)
    norm_x_tuple = tuple(num for pair in sorted(pairs.tolist()) for num in pair)

    # Fetch from cache or compute fresh (returns view/copy of 2d geometry)
    pred_base_locations, bids_pred = _evaluate_candidate_cached(norm_x_tuple, bundles)

    # Logging the trajectory
    trajectory_buffer.append(
        pred_base_locations.copy()
    )  # Copy ensures array state is frozen in time

    # Logging the proxy obj value
    proxy_objective_value = proxy_objective_func(bids_pred, bids)
    proxy_objective_buffer.append(proxy_objective_value)

    # logging true obj values
    true_objective_values = {}
    for true_objective_func in true_objective_funcs:
        true_objective_val = true_objective_func(
            pred_base_locations, _true_base_locations
        )
        true_objective_values[true_objective_func.__name__] = true_objective_val
    true_objective_buffer.append(true_objective_values)

    return proxy_objective_value


def auctioneer_optimize(
    bundles,
    bids,
    num_locations_to_estimate,
    _true_base_locations,
    opt_algorithm,
    params_lower_bounds,
    params_upper_bounds,
    rng,
    maxeval,
    proxy_objective_func=rmse,
    true_objective_funcs=[
        my_hausdorff_distance,
        my_modified_hausdorff_distance,
        my_convex_hull_jaccard_distance,
    ],
):
    "In here is where the actual optimization takes place and where I need to monitor both the proxy and the true objective function"
    num_parameters = 2 * num_locations_to_estimate
    optimizer = nlopt.opt(opt_algorithm, num_parameters)
    trajectory_buffer = []
    proxy_objective_buffer = []
    true_objective_buffer = []
    # nlopt optimizers require a function f(x, grad), so use partial to transform target_function into such
    partial_target_function = partial(
        target_function,
        bundles=bundles,
        bids=bids,
        proxy_objective_func=proxy_objective_func,
        _true_base_locations=_true_base_locations,
        true_objective_funcs=true_objective_funcs,
        trajectory_buffer=trajectory_buffer,
        proxy_objective_buffer=proxy_objective_buffer,
        true_objective_buffer=true_objective_buffer,
    )

    optimizer.set_min_objective(partial_target_function)
    optimizer.set_lower_bounds(params_lower_bounds * num_locations_to_estimate)
    optimizer.set_upper_bounds(params_upper_bounds * num_locations_to_estimate)
    optimizer.set_maxeval(maxeval)
    # optimizer.set_stopval(0)
    x0 = rng.uniform(
        params_lower_bounds, params_upper_bounds, size=(num_locations_to_estimate, 2)
    ).flatten()
    xopt = optimizer.optimize(x0)
    xopt = xopt.reshape(-1, 2, copy=True)
    opt_val = optimizer.last_optimum_value()
    return_code = optimizer.last_optimize_result()
    if return_code > 0:
        return {
            "xopt": xopt,
            "opt_val": opt_val,
            "return_code": return_code,
            "trajectory": trajectory_buffer,
            "proxy_objectives": proxy_objective_buffer,
            "true_objectives": true_objective_buffer,
        }
    else:
        raise ValueError(f"Optimization failed with code {return_code}")


def run(
    seed,
    x_min,
    x_max,
    y_min,
    y_max,
    size_auction_pool,
    num_bundles,
    true_num_locations,
    pred_num_locations,
    opt_algorithm,
    maxeval,
):
    rng = np.random.default_rng(seed)
    true_base_locations = rng.uniform(
        (x_min, y_min), (x_max, y_max), size=(true_num_locations, 2)
    )
    auction_pool = rng.uniform(
        (x_min, y_min), (x_max, y_max), size=(size_auction_pool, 2)
    )
    auction_pool = tuple(tuple(row) for row in auction_pool.tolist())
    # draw random bundles of random size
    bundles = draw_bundles(rng, size_auction_pool, num_bundles, auction_pool)
    # TODO add a depot to set_a whose location is known also to the auctioneer?
    carrier_bids = compute_bids(true_base_locations, bundles)

    # let the auctioneer use a derivative-free optimizer to reverse-engineer the true base locations using only the bidding information
    optimize_result = auctioneer_optimize(
        bundles=bundles,
        bids=carrier_bids,
        num_locations_to_estimate=pred_num_locations,
        _true_base_locations=true_base_locations,
        opt_algorithm=opt_algorithm,
        params_lower_bounds=[x_min, y_min],
        params_upper_bounds=[x_max, y_max],
        rng=rng,
        maxeval=maxeval,
        proxy_objective_func=rmse,
        true_objective_funcs=[
            my_hausdorff_distance,
            my_modified_hausdorff_distance,
            my_convex_hull_jaccard_distance,
        ],
    )

    return optimize_result


if __name__ == "__main__":
    optimize_result = run(
        rng=np.random.default_rng(1),
        x_min=0,
        x_max=100,
        y_min=0,
        y_max=100,
        size_auction_pool=12,
        num_bundles=8,
        true_num_locations=4,
        pred_num_locations=4,
        # opt_algorithm=nlopt.GN_CRS2_LM,
        opt_algorithm=nlopt.GN_DIRECT_L_RAND,
        maxeval=8,
    )
    print(optimize_result)
    print("Done.")
