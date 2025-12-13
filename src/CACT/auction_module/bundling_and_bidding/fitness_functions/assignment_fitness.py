from copy import deepcopy
from typing import Sequence, Callable

import pandas as pd

from auction_module.bundle_generation.assignment_based.assignment import Assignment
from auction_module.bundling_and_bidding.fitness_functions.bundle_fitness import BundleFitnessFunction
from auction_module.bundling_and_bidding.fitness_functions.fitness_functions import FitnessFunction
from auction_module.bundling_and_bidding.type_defs import QueriesType, ResponsesType
from auction_module.wdp import WdpGurobi
from core_module.instance import CAHDInstance
from core_module.request import Request


class AssignmentFitnessAggregateBundleFitness(FitnessFunction):
    """
    fitness = aggregation of the bidder-bundle fitness values
    if aggr is sum, then this is the estimated social welfare of the assignment

    """

    def __init__(self, bundle_fitness: BundleFitnessFunction, aggr: Callable, higher_is_better: bool = True):
        super().__init__()
        self.search_space = 'assignment'
        self._bundle_fitness: BundleFitnessFunction = bundle_fitness
        self._aggr = aggr
        self._higher_is_better = higher_is_better
        self._optimization_direction = 1 if higher_is_better else -1

        self._params = {
            'bundle_fitness': self._bundle_fitness,
            'aggr': self._aggr.__name__,
            'higher_is_better': self._higher_is_better,
        }
        self._p3_log = False
        pass

    def __repr__(self):
        prefix = '' if self._higher_is_better else '-'
        aggr_name = self._aggr.__name__ if callable(self._aggr) else self._aggr
        return f'{prefix}{aggr_name}({str(self._bundle_fitness)})'

    def __call__(self, instance, assignments: Sequence[Assignment], **kwargs):
        assignment_fitness = []
        for assignment in assignments:
            ass_fit = []
            for carrier, bundle in assignment.carrier_to_bundle().items():
                bundle_fitness = self._bundle_fitness(instance, [bundle], bidder_idx=carrier.id_)
                ass_fit.append(bundle_fitness)
            assignment_fitness.append(self._optimization_direction * self._aggr(ass_fit))
        return assignment_fitness

    def fit(self, instance: CAHDInstance, auction_request_pool: tuple[Request], queries: QueriesType,
            responses: ResponsesType):
        # if not self._p3_log:  # hacky solution for easy comparison of results
        #     self._log_p3()
        fit_results = self._bundle_fitness.fit(instance, auction_request_pool, queries, responses)
        # fit_results is a list of fit_results (i.e. metrics), one per each carrier-specific bundle fitness
        mean_fit_results = dict(pd.DataFrame(fit_results).mean())
        # NOTE do not log them to mlflow here but in bundling_and_bidding because I need to log with the correct step
        return mean_fit_results


class FitnessAssignmentWdpObjectiveIncrease(FitnessFunction):
    """
    This fitness function measures the fitness of an assignment candidate from the search space by comparing the
    objective value of (a) the optimal WDP solution considering all previously queried bundles and (b) the optimal
    WDP solution considering all previously queried bundles AND the assignment candidate.
    The responses for the assignment candidate are estimated using a bundle fitness functions, usually Neural Networks,
    or decision trees, something that directly estimates the carrier's valuation of the bundle.
    If the assignment candidate does not improve the objective value of the WDP, then the fitness is 0.
    Otherwise, the fitness is the improvement in the objective value of the WDP.

    Example:
    wdp_1 = WDP(queries, responses, sense='min')
    wdp_2 = WDP(queries + assignment_candidate, responses + estimated_assignment_candidate, sense='min')
    fitness = wdp_1.objVal - wdp_2.objVal
    """

    def __init__(self, instance: CAHDInstance, auction_request_pool: tuple[Request],
                 bundle_fitness: FitnessFunction.__class__,
                 **kwargs):
        super().__init__()
        self.search_space = 'assignment'
        assert kwargs['higher_is_better'] is True, ('The BundleFitness prediction requires higher_is_better=True,'
                                                    'since it is not used as the actual candidate fitness but just as'
                                                    'a means to estimate the fitness of the assignment candidate.')
        self.bundle_fitness = bundle_fitness(instance, auction_request_pool, **kwargs)
        self.queries = []
        self.responses = []
        self.wdp_1_objVal = None

    def __repr__(self):
        return f'FitnessAssignmentWdpObjectiveIncrease({self.bundle_fitness})'

    def __call__(self, instance, elements: Assignment, **kwargs):
        # compute optimal WDP solution considering all previously queried bundles AND the assignment candidate
        queries = deepcopy(self.queries)
        responses = deepcopy(self.responses)
        for bidder_idx, bundle in enumerate(elements):
            if any(bundle):
                queries[bidder_idx].append(bundle)
                responses[bidder_idx].append(self.bundle_fitness(bundle, bidder_idx=bidder_idx))
        wdp_2 = WdpGurobi(queries, responses, sense='min')

        # estimated WDP improvement of including the assignment candidate's queries
        fitness = self.wdp_1_objVal - wdp_2.objVal
        return fitness

    def fit(self, instance: QueriesType, auction_request_pool: tuple[Request], queries: QueriesType,
            responses: ResponsesType):
        # X_train, X_test, y_train, y_test = nested_train_test_split(queries, responses, train_size=0.8)
        self.bundle_fitness.fit(queries, auction_request_pool,
                                responses, )  # TODO validation loss in each training epoch, not just at the end
        # self.bundle_fitness.validate(X_test, y_test)
        self.queries = queries
        self.responses = responses
        # update the optimal WDP solution considering all previously queried bundles
        wdp_1 = WdpGurobi(queries, responses, sense='min')
        self.wdp_1_objVal = wdp_1.objVal
