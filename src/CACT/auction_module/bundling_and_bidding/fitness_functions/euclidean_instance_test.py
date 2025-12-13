from copy import deepcopy
from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from pyvrp.stop import MaxRuntime
from tqdm import trange

from auction_module.bundle_generation.bundle_based.bundle import Bundle
from auction_module.bundling_and_bidding.fitness_functions.vrp_learn.target_function import TargetFunction
from auction_module.bundling_and_bidding.fitness_functions.vrp_learn.optimization_policy import TrulyRandomSearch
from core_module.carrier import Carrier
from core_module.depot import Depot
from core_module.instance import CAHDInstance
from core_module.request import Request

from routing_module.routing_solver import RoutingSolver
from utility_module.io import data_dir


def vrp_learn(inst: CAHDInstance, size_auction_pool: int = 5, seed: int = 0, num_bundle_queries=10):
    rng = np.random.default_rng(seed)
    carrier_true = Carrier(0, 'Carrier 0', inst.depots[0], 'duration', 1)
    for request in inst.requests:
        if request.initial_carrier_assignment == carrier_true.id_:
            carrier_true.assign_request(request, True)

    # VRPTWMinTravelDurationInsertion ========================================
    # routing = VRPTWMinTravelDurationInsertion()
    # tours = routing.sequential_cheapest_insertion(inst, carrier_true.depot, carrier_true.accepted_requests,
    # carrier_true.max_num_tours, True)
    # fig, ax = plt.subplots()
    # for t in tours:
    #     t.plot(ax, True)
    # plt.title('VRPTWMinTravelDurationInsertion')
    # plt.show()

    # PyVrpRouting ===========================================================
    routing = PyVrpRouting()
    routing.solve_carrier(inst, carrier_true, MaxRuntime, 2)

    # select some random requests to be the auction pool
    unassigned_requests = [r for r in inst.requests if r not in carrier_true.assigned_requests]
    pool = rng.choice(unassigned_requests, size_auction_pool, replace=False)
    pool = sorted(pool, key=lambda r: r.index)

    queries_and_responses = []
    for epoch in trange(num_bundle_queries, desc='Querying bundles'):  # TODO change to a stopping criterion
        # TODO ideally the selection of the next query/queries depends on previous queries and responses as well as
        #  the current model fit. I.e., fit_carrier_model should be part of the loop
        query = select_query(pool, queries_and_responses)
        bundle = Bundle.from_binary(pool, query)
        response = ask_carrier(carrier_true, bundle, pool)
        queries_and_responses.append((query, response.total_seconds()))

    max_num_function_evaluations = 500
    optimization_policy = TrulyRandomSearch(max_num_function_evaluations=max_num_function_evaluations)
    carrier_model = fit_carrier_model(
        instance=inst,
        pool=pool,
        queries_and_responses=queries_and_responses,
        depot=carrier_true.depot,
        num_unknown_orders=5,
        max_num_tours=1,
        optimization_policy=optimization_policy,
    )
    optimization_policy.plot_history()

    carrier_model_as_carrier = carrier_model._carrier_without_bundle()
    carrier_model_instance = carrier_model._current_params_instance(carrier_model_as_carrier)
    routing.solve_carrier(carrier_model_instance, carrier_model_as_carrier, MaxRuntime, 2)

    fig, ax = plt.subplots()
    carrier_true.plot(ax)
    carrier_model_as_carrier.plot(ax)
    plt.title(f'PyVrp Routing of Carrier Model - {num_bundle_queries} queries, '
              f'{max_num_function_evaluations} evals')
    plt.xlim(0, 25)
    plt.ylim(0, 25)
    plt.show()
    return


def select_query(pool, queries_and_responses):
    # TODO these Query policies probably need an update at some point. It feels very unintuitive to not supply the pool
    #  to select from
    # bundle_query_policy = NextQueriesBundleRandom()
    # queries = [q for q, r in queries_and_responses]
    # return bundle_query_policy(None, 1, queries)

    # NOTE: temporarily using a new implementation of the random query policy
    rng = np.random.default_rng()

    query = rng.choice([0, 1], len(pool), replace=True)
    return query


def ask_carrier(carrier: Carrier, bundle: Bundle, pool: Sequence[Request]):
    without_bundles = carrier.objective

    routing = PyVrpRouting()  # TODO this should be a parameter
    tmp_carrier = deepcopy(carrier)
    for request in bundle.requests:
        tmp_carrier.assign_request(request, True)
    tmp_carrier.tours = routing.solve(inst, tmp_carrier.depot, tmp_carrier.accepted_requests,
                                      tmp_carrier.max_num_tours, MaxRuntime, 2)
    with_bundles = tmp_carrier.objective
    return with_bundles - without_bundles


def fit_carrier_model(instance: CAHDInstance,
                      pool: Sequence[Request],
                      queries_and_responses,
                      depot: Depot,
                      num_unknown_orders: int,
                      max_num_tours: int,
                      optimization_policy,
                      ):
    model_param_names = tuple(f'x{i}' for i in range(num_unknown_orders)) + tuple(
        f'y{i}' for i in range(num_unknown_orders))
    model_hparams = {
        'instance': instance,
        'auction_request_pool': pool,
        'carrier_idx': 999,
        'label': 'CarrierModel',
        'depot': depot,
        'num_unknown_orders': num_unknown_orders,
        'num_vehicles': max_num_tours,
        'routing_solver': RoutingSolver,
    }
    queries, responses = zip(*queries_and_responses)
    target_func = TargetFunction(mse, 'min', CarrierModel, queries)

    optimization_policy.set_target_function(target_func)
    best_score, best_params = optimization_policy.optimize(instance, auction_request_pool)
    best_carrier_model = CarrierModel(**model_hparams)
    best_carrier_model.current_params = best_params
    return best_carrier_model


if __name__ == '__main__':
    inst = CAHDInstance.from_json(
        data_dir.joinpath('CR_AHD_instances/euclidean_instances/t=euclidean+d=7+c=3+n=10+v=3+o=100+r=01.json'))
    inst.plot()
    plt.show()
    vrp_learn(inst, size_auction_pool=10, num_bundle_queries=10)
    pass
