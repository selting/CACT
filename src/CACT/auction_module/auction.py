import datetime as dt
import logging
import warnings
from copy import deepcopy
from typing import Sequence, Optional, Any

import mlflow
import pandas as pd
from gurobipy import GurobiError

from auction_module.bundle_generation.assignment_based.assignment import Assignment
from auction_module.bundle_generation.bundle_based import bundle as bdl
from auction_module.bundle_generation.partition_based import partition as prt
from auction_module.bundling_and_bidding.bundling_and_bidding import BundlingAndBidding
from auction_module.wdp import WdpGurobi, WdpPyomo
from core_module import instance as it, solution as slt
from core_module.instance import CAHDInstance
from utility_module.parameterized_class import ParameterizedClass, my_value_parser, flatten_dict
from core_module.request import Request
from core_module.solution import CAHDSolution
from utility_module import profiling as pr, utils as ut
from utility_module.datetime_handling import safe_divide_timedelta

logger = logging.getLogger(__name__)


class Auction(ParameterizedClass):
    def __init__(self, num_submitted_requests: int, bundling_and_bidding: BundlingAndBidding):
        """
        Auction class can be called with various parameters to create different auction variations

        :param num_submitted_requests: number (or fraction) of requests that each carrier must release to the auction
        for re-allocation
        """
        # self.id_ = instance.id_
        # self.meta = instance.meta
        # self.solver_config = dict()

        self.num_submitted_requests = num_submitted_requests
        self.bundling_and_bidding: BundlingAndBidding = bundling_and_bidding

        self.auction_request_pool: Optional[tuple[Request]] = None
        self.original_assignment: Optional[Assignment] = None
        self.bundle_bidding_jobs: Optional[pd.DataFrame] = None
        self.auction_bundle_pool: Optional[Sequence[bdl.Bundle]] = None
        self.auction_partition_pool: Optional[Sequence[prt.Partition]] = None
        self.bids_matrix: Optional[pd.DataFrame] = None  # usually as dt.timedelta
        self.num_inf_bids: Optional[int] = None
        self.CAP_num_feasible_sol: Optional[int] = None
        self.num_feas_bids: Optional[int] = None
        self.rel_num_inf_bids: Optional[float] = None
        self.rel_num_feas_bids: Optional[float] = None
        self.winner_assignment: Optional[Assignment] = None

        self.isolated_solution_values = None
        self.collaborative_solution_values = None
        self.central_solution_values = None
        self.collaboration_gains = dict()
        self.bundle_pool_bundle_features = None
        self.bundle_pool_bundle_carrier_features = None
        self.partition_pool_metrics = None

        self._params: dict[str, Any] = {
            'num_submitted_requests': num_submitted_requests,
            'bundling_and_bidding': bundling_and_bidding,
        }
        self.metrics = {

        }

    def __repr__(self):
        s = f'Auction  '
        s += f'for {len(self.auction_request_pool)} requests' if self.auction_request_pool else ''
        return s

    @property
    def original_bundles(self):
        """

        :return: the submitted requests, bundled by carrier.
        """
        if self.auction_request_pool and self.original_assignment:
            return self.original_assignment.bundles()
        else:
            return None

    @property
    def original_bundle_bids(self):
        """
        :return: a dict that maps each carrier_id to the tuple of (bundle, bid) for the bundle that the carrier
        originally submitted
        """
        original_bundle_bids = dict()
        for carrier_id, bundle in self.original_assignment.carrier_to_bundle().items():
            bid_idx = self.auction_bundle_pool[carrier_id].index(bundle.bitstring)
            original_bundle_bids[carrier_id] = (bundle, self.bids_matrix[carrier_id][bid_idx])
        return original_bundle_bids

    @property
    def winner_bundles(self) -> tuple[bdl.Bundle]:
        """

        :return: the bundles that got reallocated, ordered by carrier. May include empty
        Sequences for carriers that did not win any bundle.
        """
        if self.auction_request_pool and self.winner_assignment:
            return self.winner_assignment.bundles()
        else:
            return None

    @property
    def winner_bundle_bids(self) -> Optional[dict[int, tuple[bdl.Bundle, dt.timedelta]]]:
        """
        :return: a dict that maps each carrier_id to the tuple of (bundle, bid) that this carrier won
        """
        if self.bids_matrix is not None:
            winner_bundle_bids = dict()
            for carrier_id, bundle in self.winner_assignment.carrier_to_bundle().items():
                if sum(bundle.bitstring) == 0:
                    winner_bundle_bids[carrier_id] = (bundle, 0)
                else:
                    bid_idx = self.auction_bundle_pool[carrier_id].index(bundle.bitstring)
                    winner_bundle_bids[carrier_id] = (bundle, self.bids_matrix[carrier_id][bid_idx])
            return winner_bundle_bids
        else:
            return None

    @property
    def degree_of_reallocation(self):
        if self.original_assignment and self.winner_assignment:

            hamming_dist = ut.hamming_distance(tuple(self.original_assignment.values()),
                                               tuple(self.winner_assignment[k] for k in self.original_assignment))
            degree_of_reallocation = hamming_dist / len(self.original_assignment)
            return degree_of_reallocation
        else:
            return None

    def log_params(self):
        mlflow.log_params(flatten_dict(self.params, 'auction', '__'))

    def run_auction(self, instance: it.CAHDInstance, solution: slt.CAHDSolution):
        # SETUP
        pre_auction_solution = deepcopy(solution)
        logger.debug(f'running auction {self.__class__.__name__}')
        self.log_params()

        # REQUEST SELECTION
        original_assignment, auction_request_pool = self.execute_request_selection(instance, solution)
        self.original_assignment, self.auction_request_pool = original_assignment, auction_request_pool

        # DETERMINE THE NEW ALLOCATION
        assert self.auction_request_pool, f'No requests have been submitted!'
        queries, responses = self.bundling_and_bidding(instance, solution, auction_request_pool, original_assignment)

        # WINNER DETERMINATION
        try:
            wdp = WdpGurobi(queries, responses, sense='min')  # minimize the sum of realized routing costs
        except GurobiError as e:
            warnings.warn(f'WDP: Gurobi error: {e}, resorting to pyomo with CBC solver')
            wdp = WdpPyomo(queries, responses, sense='min', solver_name='cbc')

        item_allocation = wdp.item_allocation
        mlflow.log_metric('runtime_wdp', wdp.runtime)

        # GENERATE THE WINNER ASSIGNMENT
        self.winner_assignment = Assignment(solution.carriers)
        for request_idx, request in enumerate(self.auction_request_pool):
            for carrier in solution.carriers:
                if item_allocation[carrier.id_][request_idx] == 1:
                    self.winner_assignment[request] = carrier
                    break
        self.bids_matrix = responses

        # ASSIGN BUNDLES AND ROUTE THEIR REQUESTS
        for request, carrier in self.winner_assignment.items():
            carrier.assign_request(request)
            carrier.accept_request(request)
        for carrier in solution.carriers:
            bundle = self.winner_assignment.carrier_to_bundle()[carrier]
            carrier.route_new_bundle(instance, bundle)

        # CATCH ERRORS
        if solution.objective > pre_auction_solution.objective:
            raise ValueError(f'{instance.id_},:\n'
                             f' Post={solution.objective}; Pre={pre_auction_solution.objective}\n'
                             f' Post-auction objective is worse than pre-auction objective!\n'
                             f' Recovering the pre-auction solution.')
            solution = pre_auction_solution
            assert pre_auction_solution.objective == solution.objective

        auction_metrics = self.compute_auction_metrics(responses, pre_auction_solution, solution)
        mlflow.log_metrics({k: my_value_parser(v, True) for k, v in auction_metrics.items()})

        return solution

    def compute_auction_metrics(self, responses, pre_auction_solution: CAHDSolution,
                                post_auction_solution: CAHDSolution) -> dict[str, float]:
        """
        Computes the metrics for the auction. The metrics are stored in the self.metrics dictionary

        Parameters
        ----------
        responses : Sequence[Sequence[dt.timedelta]] : the responses of the carriers
        pre_auction_solution : the solution before the auction
        post_auction_solution : the solution after the auction

        Returns
        -------
        None

        """
        metrics = {
            'num_bids': sum(len(r) for r in responses),
            'num_inf_bids': sum(1 for r in responses for b in r if b == 'infeasible'),
            'num_feas_bids': sum(1 for r in responses for b in r if b != 'infeasible'),
        }
        metrics['rel_num_inf_bids'] = metrics['num_inf_bids'] / metrics['num_bids']
        metrics['rel_num_feas_bids'] = metrics['num_feas_bids'] / metrics['num_bids']
        # number of bundles for which ALL carriers submitted an infeasible bid
        # num_inf_bundles = (self.bids_matrix == dt.timedelta.max).all(axis=1).sum()

        num_reallocated_orders = ut.hamming_distance(tuple(self.original_assignment.values()),
                                                     tuple(self.winner_assignment[k] for k in self.original_assignment))
        metrics['degree_of_reallocation'] = num_reallocated_orders / len(self.original_assignment)

        pre_df = pd.Series(pre_auction_solution.metrics)
        post_df = pd.Series(post_auction_solution.metrics)
        gain_df = pd.DataFrame(post_df - pre_df, columns=['gain'])
        if num_reallocated_orders > 0:
            gain_df['gain_per_reallocation'] = gain_df['gain'] / num_reallocated_orders
            gain_df['rel_gain'] = safe_divide_timedelta(gain_df['gain'], pre_df)
            gain_df['rel_gain_per_reallocation'] = gain_df['rel_gain'] / num_reallocated_orders

        else:
            assert all(pre_df == post_df), \
                'No reallocation, but post-auction solution is different from pre-auction solution'
            gain_df['gain_per_reallocation'] = gain_df['gain']  # if nothing reallocated > gain=0 > gain_per_realloc=0
            gain_df['rel_gain'] = 0.0
            gain_df['rel_gain_per_reallocation'] = 0.0

        for col in gain_df.columns:
            for row in gain_df.index:
                metrics[f"{col}_{row}"] = gain_df.at[row, col]

        return metrics

    def on_after_auction(self, instance: CAHDInstance, solution: CAHDSolution):
        if not self.original_assignment.keys() == self.winner_assignment.keys():
            raise KeyError(
                f'Original and winner assignment do not include the same requests!\n'
                f'original:\n{self.original_assignment}'
                f'winner:\n{self.winner_assignment}\n'
                f'instance:\n{instance.id_}\n')
        pass

    def execute_request_selection(self, instance: CAHDInstance, solution: CAHDSolution) -> (Assignment, tuple[Request]):

        if isinstance(self.num_submitted_requests, int):
            k = self.num_submitted_requests
        elif isinstance(self.num_submitted_requests, float):
            if self.num_submitted_requests % 1 == 0:
                k = int(self.num_submitted_requests)
            else:
                if not self.num_submitted_requests <= 1:
                    raise ValueError('If providing a float, must be <= 1 to be converted to percent')
                k = round((instance.num_requests / instance.num_carriers) * self.num_submitted_requests)
        else:
            raise ValueError

        timer = pr.Timer()
        original_assignment = Assignment(solution.carriers)
        for carrier in solution.carriers:
            selected_requests = carrier.select_requests_for_auction(instance, k)
            for request in selected_requests:
                carrier.release_requests(instance, [request])
                original_assignment[request] = carrier
        auction_request_pool = tuple(sorted(original_assignment.requests()))
        timer.stop()
        mlflow.log_metric('runtime_request_selection', timer.duration)

        if self.auction_request_pool:
            logger.debug(f'requests {self.auction_request_pool} have been submitted to the auction pool')
        return original_assignment, auction_request_pool

    # def to_json(self):
    #     path = io.unique_path(io.auctions_dir, self.id_ + '_#{:03d}' + '.json')
    #     # if sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) >= 5e+8:  # 500 MB limit
    #     # delete some old files
    #
    #     with open(path, mode='w') as f:
    #         json.dump(self.summary(), f, sort_keys=False, indent=4, cls=io.MyJSONEncoder)
    #     pass

    def compute_collaboration_gains(self,
                                    isolated_solution: slt.CAHDSolution,
                                    collaborative_solution: slt.CAHDSolution,
                                    central_solution: Optional[slt.CAHDSolution] = None):
        """
        computes collaboration gains as the percentage of improvement over the isolated planning approach.
        Also computes these values relative to the number of exchanged requests

        :param isolated_solution:
        :param collaborative_solution:
        :param central_solution:
        :return:
        """
        assert isolated_solution.solver_config['solution_algorithm'] == 'IsolatedPlanning'
        assert collaborative_solution.solver_config['solution_algorithm'] == 'CollaborativePlanning'

        assert all(isolated_solution.solver_config[k] == collaborative_solution.solver_config[k] ==
                   self.solver_config[k] for k in general_config)
        assert all(collaborative_solution.solver_config[k] == self.solver_config[k]
                   for k in collaborative_config)

        if central_solution:
            assert central_solution.solver_config['solution_algorithm'] == 'CentralPlanning'
            assert all(central_solution.solver_config[k] == self.solver_config[k] for k in general_config)

        solution_values = [
            'objective',
            'sum_travel_distance',
            'sum_travel_duration',
            'sum_wait_duration',
            'sum_service_duration',
            'sum_idle_duration',
            'sum_revenue',
            'utilization',
            'num_tours',
            'num_pendulum_tours',
            'num_routing_stops',
            'acceptance_rate',
        ]

        isolated_summary = isolated_solution.summary()
        self.isolated_solution_values = {'isolated_' + k: isolated_summary[k] for k in solution_values}
        collaborative_summary = collaborative_solution.summary()
        self.collaborative_solution_values = {'collaborative_' + k: collaborative_summary[k] for k in solution_values}
        if central_solution:
            central_summary = central_solution.summary()
            self.central_solution_values = {'central_' + k: central_summary[k] for k in solution_values}

        nrr = ut.hamming_distance(tuple(self.original_assignment.values()),
                                  tuple(self.winner_assignment[k] for k in self.original_assignment))

        # rel_savings: collaboration savings as a fraction of the isolated planning
        # rel_savings_per_reallocation: relative collaboration gain, i.e., rel_savings relative to the number fo reallocated requests
        # abs_savings_potential: collaboration gain potential, i.e., the maximum possible gain as defined by the central solution
        # abs_savings_potential_achieved: collaboration gain potential achieved, i.e., how much of the potential was exhausted (rel_savings/abs_savings_potential)
        for k in solution_values:
            # gain is the absolute amount obtained by the collaborative solution
            if isolated_summary[k] == dt.timedelta(0):
                abs_savings = collaborative_summary[k].total_seconds()
                abs_savings_per_reallocation = (abs_savings / nrr) if nrr > 0 else 0
                rel_savings = None
                rel_savings_per_reallocation = None
                if central_solution:
                    abs_savings_potential = central_summary[k].total_seconds()
                    if abs_savings_potential != 0:
                        abs_savings_potential_achieved = abs_savings / abs_savings_potential
                    else:
                        abs_savings_potential_achieved = 0

            # gain is the absolute amount obtained by the collaborative solution
            elif isolated_summary[k] == 0:
                abs_savings = collaborative_summary[k]
                abs_savings_per_reallocation = (abs_savings / nrr) if nrr > 0 else 0
                rel_savings = None
                rel_savings_per_reallocation = None
                if central_solution:
                    abs_savings_potential = central_summary[k]
                    if abs_savings_potential != 0:
                        abs_savings_potential_achieved = abs_savings / abs_savings_potential
                    else:
                        abs_savings_potential_achieved = 0

            # iff the isolated solution has a proper value we can compute the gain relative to that value
            else:
                abs_savings = isolated_summary[k] - collaborative_summary[k]
                if isinstance(abs_savings, dt.timedelta):
                    abs_savings = abs_savings.total_seconds()
                abs_savings_per_reallocation = (abs_savings / nrr) if nrr > 0 else 0
                rel_savings = 1 - (collaborative_summary[k] / isolated_summary[k])
                rel_savings_per_reallocation = (rel_savings / nrr) if nrr > 0 else 0
                if central_solution:
                    abs_savings_potential = central_summary[k]
                    rel_savings_potential = 1 - (central_summary[k] / isolated_summary[k])
                    if abs_savings_potential != 0:
                        abs_savings_potential_achieved = abs_savings / abs_savings_potential
                        rel_savings_potential_achieved = rel_savings / rel_savings_potential
                    else:
                        abs_savings_potential_achieved = None
                        rel_savings_potential_achieved = None

            self.collaboration_gains['abs_savings_' + k] = abs_savings
            self.collaboration_gains['abs_savings_per_reallocation_' + k] = abs_savings_per_reallocation
            self.collaboration_gains['rel_savings_' + k] = rel_savings
            self.collaboration_gains['rel_savings_per_reallocation_' + k] = rel_savings_per_reallocation
            if central_solution:
                self.collaboration_gains['abs_savings_potential_' + k] = abs_savings_potential
                self.collaboration_gains['rel_savings_potential_' + k] = rel_savings_potential
                self.collaboration_gains['abs_savings_potential_achieved_' + k] = abs_savings_potential_achieved
                self.collaboration_gains['rel_savings_potential_achieved_' + k] = rel_savings_potential_achieved
        pass
