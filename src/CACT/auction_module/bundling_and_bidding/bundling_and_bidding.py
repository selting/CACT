from typing import Any

import mlflow
import pandas as pd
from tqdm import tqdm

from auction_module.bundle_generation.assignment_based.assignment import Assignment
from auction_module.bundling_and_bidding.fitness_functions.fitness_functions import FitnessFunction
from auction_module.bundling_and_bidding.fitness_functions.vrp_learn.bundle_fitness_carrier_model import \
    BundleFitnessCarrierModel
from auction_module.bundling_and_bidding.fitness_functions.vrp_learn.distance import *
from auction_module.bundling_and_bidding.fitness_functions.vrp_learn.distance_test import plot_cases
from auction_module.bundling_and_bidding.next_queries import NextQueries
from auction_module.bundling_and_bidding.type_defs import QueriesType, ResponsesType
from core_module.carrier import Carrier
from core_module.instance import CAHDInstance
from core_module.request import Request
from core_module.solution import CAHDSolution
from utility_module.parameterized_class import ParameterizedClass
from utility_module.profiling import track_cumulative_time
import matplotlib.pyplot as plt


class BundlingAndBidding(ParameterizedClass):
    def __init__(self, fitness_function: FitnessFunction, next_queries: NextQueries, num_bidding_jobs: int,
                 num_initial_queries: int, num_queries_per_query_epoch: int):
        if num_initial_queries + num_queries_per_query_epoch > num_bidding_jobs:
            raise ValueError('The number of initial and iterative queries must be less than or equal to the number'
                             ' of bidding jobs, i.e. the total number of queries.')
        self._fitness_function: FitnessFunction = fitness_function
        self._next_queries: NextQueries = next_queries
        self.num_bidding_jobs = num_bidding_jobs
        self.num_initial_queries = num_initial_queries
        self.num_queries_per_query_epoch = num_queries_per_query_epoch

        if next_queries.__class__.__name__.startswith('NextQueriesBundle'):
            self.search_space = 'Bundles'
        elif next_queries.__class__.__name__.startswith('NextQueriesPartition'):
            self.search_space = 'Partitions'
        elif next_queries.__class__.__name__.startswith('NextQueriesAssignment'):
            self.search_space = 'Assignments'
        else:
            raise ValueError(f'Cannot extract search space from NextQueries type: {next_queries.__name__}')

        self._params: dict[str, Any] = {
            'fitness_function': self._fitness_function,
            'next_queries': self._next_queries,
            'num_bidding_jobs': self.num_bidding_jobs,
            'num_initial_queries': self.num_initial_queries,
            'num_queries_per_query_epoch': self.num_queries_per_query_epoch,
        }

        pass

    def __repr__(self):
        return f'{self.__class__.__name__}({self.params})'

    def __str__(self):
        return f'{self.__class__.__name__}'

    def __call__(self, instance: CAHDInstance, solution: CAHDSolution, auction_request_pool: tuple[Request],
                 original_assignment: Assignment) -> tuple[QueriesType, ResponsesType]:
        # INITIAL PHASE
        # query num_initial_queries bundles for each carrier, always including the original bundle
        queries: QueriesType = self._initial_queries(solution, auction_request_pool, original_assignment)
        responses: ResponsesType = self.get_responses(instance, solution, queries)
        fit_queries, fit_responses = self._select_valid_training_data(queries, responses)
        mean_fit_results = self._fitness_function.fit(instance, auction_request_pool, fit_queries, fit_responses)
        mlflow.log_metrics(mean_fit_results, step=0)  # logs the average mse, mape, rmse, ... across all carriers
        self._log_carrier_model_evaluation(solution, t=0)

        # ITERATIVE PHASE (optional, only if self.num_initial_queries < self.num_bidding_jobs)
        t = 1
        while any(len(x) < self.num_bidding_jobs for x in responses):
            adj_num_queries = self._get_adj_num_queries(queries)
            current_queries = self._next_queries(instance, solution.carriers, auction_request_pool,
                                                 self._fitness_function, adj_num_queries, queries)
            current_responses = self.get_responses(instance, solution, current_queries)

            for bidder_idx, (q, r) in enumerate(zip(current_queries, current_responses)):
                queries[bidder_idx].extend(q)
                responses[bidder_idx].extend(r)

            fit_queries, fit_responses = self._select_valid_training_data(queries, responses)
            mean_fit_results = self._fitness_function.fit(instance, auction_request_pool, fit_queries, fit_responses)
            mlflow.log_metrics(mean_fit_results, t)
            self._log_carrier_model_evaluation(solution, t)
            # self.plot_carriers_vs_estimated(instance, solution)

            t += 1

        # logging
        # mlflow.log_metric('runtime_next_queries', self._next_queries.__call__.cumulative_times['__call__'])
        # mlflow.log_metric('runtime_get_responses', self.get_responses.cumulative_times['get_responses'])

        return queries, responses

    def _initial_queries(self, solution: CAHDSolution, auction_request_pool: tuple[Request],
                         original_assignment: Assignment) -> QueriesType:
        """
        Select the initial queries by first including the original bundle as a query, and then filling up the rest
        from random Assignments.
        This might change in the future to apply a more sophisticated initial sampling strategy.
        """
        initial_queries = []
        # first, include the original bundle as a query
        for carrier in solution.carriers:
            initial_queries.append([original_assignment.carrier_to_bundle()[carrier]])

        # then, fill up the rest of the queries by generating random assignments
        while any(len(x) < self.num_initial_queries for x in initial_queries):
            assignment = Assignment.random(solution.carriers, auction_request_pool)
            for carrier, bundle in assignment.carrier_to_bundle().items():
                if any(bundle.requests):  # do not query the empty bundle
                    if len(initial_queries[carrier.id_]) < self.num_initial_queries:
                        if bundle not in initial_queries[carrier.id_]:
                            initial_queries[carrier.id_].append(bundle)
        # safety check
        for iq in initial_queries:
            if len(iq) != self.num_initial_queries:
                raise ValueError('The number of initial queries is not as expected.')
        return initial_queries

    def _log_carrier_model_evaluation(self, solution, t):
        """
        Log the carrier model evaluation metrics to mlflow.
        Really just a workaround because the BundleFitnessCarrierModel.fit() method does not have access to the true
        coordinates.
        """
        if hasattr(self._fitness_function, '_bundle_fitness'):
            if isinstance(self._fitness_function._bundle_fitness, BundleFitnessCarrierModel):
                # TODO these should be provided in the config somewhere, not hardcoded here
                all_distance_funcs = [my_hausdorff_distance,
                                      my_modified_hausdorff_distance,
                                      my_convex_hull_jaccard_distance,
                                      my_tsp_hull_jaccard_distance,
                                      my_tsp_obj_val_diff,
                                      my_MinWBMP,
                                      ]
                records = []
                cases = []
                # for carrier, carrier_model in zip(solution.carriers, self._fitness_function._bundle_fitness._models):
                for carrier_idx in range(len(solution.carriers)):
                    carrier = solution.carriers[carrier_idx]
                    carrier_model = self._fitness_function._bundle_fitness._models[carrier_idx]
                    A = np.array([(r.x, r.y) for r in carrier.accepted_requests])
                    B = np.array([(carrier_model.current_params[f'x{i}'], carrier_model.current_params[f'y{i}'])
                                  for i in range(len(carrier_model.current_params) // 2)])
                    cases.append((A, B))
                    record = {'carrier': carrier.id_}
                    for dist_func in all_distance_funcs:
                        dist_value = dist_func(A, B)
                        mlflow.log_metric(f'{dist_func.__name__}_{carrier_idx}', dist_value)
                        # print(f'{dist_func.__name__}({carrier.id_}) = {dist_value}')
                    records.append(record)
                # log the average of the distance across carriers & the distance measure figure
                dist_df = pd.DataFrame.from_records(records, index='carrier')
                mean_df = dist_df.mean(numeric_only=True, axis=0)
                mlflow.log_metrics(dict(mean_df), timestamp=t)
                # fig = plot_cases(cases, all_distance_funcs)
                # mlflow.log_figure(fig, artifact_file="distance_measures.png")

    def _get_adj_num_queries(self, queries):
        min_num_queries = min(len(queries[i]) for i in range(len(queries)))
        adj_num_queries = min(self.num_queries_per_query_epoch, self.num_bidding_jobs - min_num_queries)
        return adj_num_queries

    def _select_valid_training_data(self, queries, responses):
        # only train with the queries that have a valid response
        fit_queries, fit_responses = [], []
        for carrier_queries, carrier_responses in zip(queries, responses):
            fit_queries_sub, fit_responses_sub = [], []
            for query, response in zip(carrier_queries, carrier_responses):
                if response != 'infeasible':
                    fit_queries_sub.append(query)
                    fit_responses_sub.append(response)
            fit_queries.append(fit_queries_sub)
            fit_responses.append(fit_responses_sub)
        return fit_queries, fit_responses

    @track_cumulative_time
    def get_responses(self, instance: CAHDInstance, solution: CAHDSolution, queries: QueriesType) -> ResponsesType:
        all_responses = []
        pbar = tqdm(total=sum(len(x) for x in queries), desc='Computing bids on bundles', disable=True)
        for carrier_idx, carrier in enumerate(solution.carriers):
            carrier_responses = []
            for bundle in queries[carrier_idx]:
                assert any(bundle.requests), ('Querying the empty bundle is not allowed as it can cause the '
                                              'post-auction result to be worse than the pre-auction result.')
                response = carrier.compute_bid_on_bundle(instance, bundle)
                carrier_responses.append(response)
                pbar.update()
            all_responses.append([x for x in carrier_responses])
        return all_responses

    def plot_carriers_vs_estimated(self, instance: CAHDInstance, solution: CAHDSolution):
        if not isinstance(self._fitness_function._bundle_fitness, BundleFitnessCarrierModel):
            return
        else:
            self._fitness_function: BundleFitnessCarrierModel
            plt.style.use('seaborn-v0_8')
            fig, axs = plt.subplots(len(solution.carriers), 1, sharex=True, sharey=True, figsize=(6, 12))
            fig: plt.Figure
            for idx in range(len(solution.carriers)):
                carrier_true: Carrier = solution.carriers[idx]

                carrier_model = self._fitness_function._bundle_fitness._models[idx]
                carrier_est: Carrier = carrier_model._carrier_without_bundle(instance)
                current_params_instance = carrier_model._current_params_instance(instance)
                carrier_est.route_all_accepted_statically(current_params_instance)

                ax: plt.Axes = axs[idx]
                carrier_true.plot(ax)
                carrier_est.plot(ax)
                ax.legend([None, None, 'True', None, None, 'Estimated'])
            opt_policy = self._fitness_function._bundle_fitness._optimization_policy
            fig.suptitle(
                f'{opt_policy.__class__.__name__}: max. {opt_policy.max_num_function_evaluations} function evaluations')
            plt.show()
