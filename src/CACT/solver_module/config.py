import datetime as dt
import itertools
from typing import Sequence, Union, Any, Generator

import nlopt
import numpy as np
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)

import auction_module.bundling_and_bidding.fitness_functions.assignment_fitness as ffa
from auction_module.auction import Auction
from auction_module.bundling_and_bidding import next_queries as nq
from auction_module.bundling_and_bidding.bundling_and_bidding import BundlingAndBidding
from auction_module.bundling_and_bidding.fitness_functions.vrp_learn.bundle_fitness_carrier_model import (
    BundleFitnessCarrierModel,
)
from auction_module.bundling_and_bidding.fitness_functions.vrp_learn.optimization_policy import (
    TrulyRandomSearch,
    Nlopt,
    BayesianOptimizationPolicy,
)
from auction_module.request_selection import (
    request_selection as rs,
    request_based_rs as rsi,
)
from routing_module.bundle_insertion import (
    StaticSequentialCheapestInsertionBundleInsertion,
)
from routing_module.bundle_insertion_feasibility_check import (
    StaticSequentialCheapestInsertionBIFC,
    disclosure_time_key,
)
from routing_module.dynamic_routing import DynamicCheapestInsertion
from routing_module.insertion_criterion import MinDuration
from routing_module.request_insertion_feasibility_check import SimpleInsertionRIFC
from routing_module.routing_solver import RoutingSolver
from routing_module.static_routing import StaticSequentialInsertion
from solver_module.solver import CollaborativeSolver
from tw_management_module import tw_offering as two, tw_selection as tws
from utility_module.io import get_git_hash


def configs() -> Generator[CollaborativeSolver, Any, None]:
    """
    generate dicts with all parameters to be tested
    """

    s_routing_solver: Sequence = [
        RoutingSolver(
            request_insertion_feasibility_check=SimpleInsertionRIFC(),
            request_insertion=DynamicCheapestInsertion(MinDuration()),
            bundle_insertion_feasibility_check=StaticSequentialCheapestInsertionBIFC(
                disclosure_time_key
            ),
            bundle_insertion=StaticSequentialCheapestInsertionBundleInsertion(
                disclosure_time_key, MinDuration()
            ),
            static_routing=None,  # not required here
        ),
    ]

    s_time_window_length: Sequence[dt.timedelta] = [
        # dt.timedelta(hours=0.5),
        # dt.timedelta(hours=1),
        # dt.timedelta(hours=2),
        # dt.timedelta(hours=4),
        dt.timedelta(hours=8),
    ]

    s_time_window_offering: list[two.TWOfferingBehavior.__class__] = [
        two.FeasibleTW,
    ]

    s_time_window_selection: Sequence[tws.TWSelectionBehavior] = [
        tws.UnequalPreference(),
        # tws.UniformPreference(),
    ]

    s_num_submitted_requests: Sequence[Union[int, float]] = [
        # 0.1,
        # 0.2,
        0.4
    ]

    s_request_selection: Sequence[rs.RequestSelectionStrategy] = [
        # rsi.Random(),
        rsi.MarginalTravelDurationProxy(),
    ]

    GA_params = {
        "num_generations": 2,  # FIXME: set back to 25
        "population_size": 250,  # FIXME: set back to 250
        "crossover_rate": 0.8,
        "mutation_probability": 0.05,
        "elitism": 2,
        "logging": False,
    }

    s_bundling_and_bidding: Sequence = (
        [
            # BundlingAndBidding(
            #     fitness_function=ffa.AssignmentFitnessAggregateBundleFitness(
            #         bundle_fitness=ffb.BundleFitnessLinearRegression(
            #             interaction_degree=1,
            #             higher_is_better=True,
            #             metrics=[mean_absolute_percentage_error]),
            #         aggr=np.mean,
            #         higher_is_better=False),
            #     # next_queries=nq.NextQueriesAssignmentGeneticAlgorithm(**GA_params),
            #     next_queries=nq.NextQueriesAssignmentRandom(),
            #     num_queries_per_query_epoch=num_queries_per_query_epoch
            # ),
        ]
        + [
            BundlingAndBidding(
                fitness_function=ffa.AssignmentFitnessAggregateBundleFitness(
                    bundle_fitness=BundleFitnessCarrierModel(
                        higher_is_better=False,
                        optimization_policy=optimization_policy,
                        error_function=error_function,
                        num_unknown_orders=6,  # TODO make this variable?
                        num_vehicles=1,  # TODO make this variable?
                        routing_solver=RoutingSolver(
                            request_insertion_feasibility_check=SimpleInsertionRIFC(),
                            request_insertion=DynamicCheapestInsertion(MinDuration()),
                            bundle_insertion_feasibility_check=StaticSequentialCheapestInsertionBIFC(
                                disclosure_time_key
                            ),
                            bundle_insertion=StaticSequentialCheapestInsertionBundleInsertion(
                                disclosure_time_key, MinDuration()
                            ),
                            static_routing=StaticSequentialInsertion(),
                        ),
                        prediction_metrics=[
                            mean_absolute_percentage_error,
                            root_mean_squared_error,
                            r2_score,
                            mean_squared_error,
                        ],
                    ),
                    aggr=np.sum,
                    higher_is_better=False,
                ),
                next_queries=nq.NextQueriesAssignmentRandom(),
                num_bidding_jobs=num_bidding_jobs,
                num_initial_queries=num_bidding_jobs,  # query all queries at the beginning, no iterative querying
                num_queries_per_query_epoch=0,
            )
            for optimization_policy in [
                opt_policy(max_num_function_evaluations, **hyperparams)
                for opt_policy, hyperparams in [
                    (TrulyRandomSearch, {}),
                    # (BayesianOptimizationPolicy, {}),
                ]
                + [
                    (
                        Nlopt,
                        {
                            "algorithm": algo,
                            "lexicographic_ordering_constraint": lexicographic_order,
                        },
                    )
                    for algo, lexicographic_order in [
                        # (nlopt.GN_DIRECT_L, False),
                        # (nlopt.GN_DIRECT, False),
                        # (nlopt.GN_DIRECT_L_RAND, False),
                        #
                        # (nlopt.GN_CRS2_LM, False),
                        #
                        # # see https://nlopt.readthedocs.io/en/latest/NLopt_Introduction/#termination-tests-for-global-optimization for MLSL stopping criterion
                        # # requires setting a local optimizer -> not yet implemented
                        # (nlopt.GN_MLSL, False),
                        # (nlopt.GN_MLSL_LDS, False),
                        #
                        # (nlopt.GN_ISRES, False),
                        # (nlopt.GN_ISRES, True),
                        #
                        # (nlopt.GN_ESCH, False),
                        #
                        # # (nlopt.GN_AGS, False),  # works only with up to 10 dimensions
                        # # (nlopt.GN_AGS, True),  # works only with up to 10 dimensions
                        #
                        # (nlopt.LN_COBYLA, False),
                        # (nlopt.LN_COBYLA, True),
                        #
                        # (nlopt.LN_BOBYQA, False),
                        #
                        # (nlopt.LN_NEWUOA_BOUND, False),  # superseded by BOBYQA, but better than bobyqa in my results
                        #
                        # (nlopt.LN_PRAXIS, False),  # performed poorly before
                        #
                        # (nlopt.LN_NELDERMEAD, False),
                        #
                        # (nlopt.LN_SBPLX, False),
                    ]
                ]
                for max_num_function_evaluations in [
                    256,
                ]
            ]
            for error_function in [
                # MAPE is undefined for y_true = 0. (leads to unexpected results, see sklearn implementation). Happens with
                #  data__r=3 for carrier 0
                # mean_absolute_percentage_error,
                # mean_squared_error,
                root_mean_squared_error,
                # r2_score,
            ]
            for num_bidding_jobs in [
                16,
                # 32,
                # 64,
                # 128,
            ]
        ]
    )

    # print(f'The current git hash is: {get_git_hash()}')

    # ===== Nested Parameter Loops =====
    for (
        routing_solver,
        time_window_length,
        time_window_offering,
        time_window_selection,
    ) in itertools.product(
        s_routing_solver,
        s_time_window_length,
        s_time_window_offering,
        s_time_window_selection,
    ):
        # isolated_planning = IsolatedSolver(
        #     routing_solver=routing_solver,
        #     time_window_selection=time_window_selection,
        #     time_window_length=time_window_length,
        # )
        # yield isolated_planning

        # central_planning = CentralSolverPyVrp(
        #     request_acceptance=request_acceptance,
        #     tour_construction=tour_construction,
        #     tour_improvement=tour_improvement,
        #     max_runtime=10,
        # )
        # yield central_planning

        # auction-specific parameters
        for num_submitted_requests in s_num_submitted_requests:
            for request_selection in s_request_selection:
                for bundling_and_bidding in s_bundling_and_bidding:
                    auction = Auction(
                        num_submitted_requests=num_submitted_requests,
                        bundling_and_bidding=bundling_and_bidding,
                    )

                    collaborative_planning = CollaborativeSolver(
                        routing_solver=routing_solver,
                        time_window_selection=time_window_selection,
                        time_window_length=time_window_length,
                        request_selection_strategy=request_selection,
                        auction=auction,
                    )

                    yield collaborative_planning
