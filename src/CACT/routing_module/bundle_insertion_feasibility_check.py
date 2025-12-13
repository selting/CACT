import abc

import pyvrp
from pyvrp.stop import MultipleCriteria, MaxRuntime, NoImprovement

from auction_module.bundle_generation.bundle_based.bundle import Bundle
from core_module.depot import Depot
from core_module.instance import CAHDInstance
from core_module.tour import Tour
from utility_module.parameterized_class import ParameterizedClass
from .dynamic_routing import DynamicCheapestInsertion
from .insertion_criterion import MinDuration
from .pyvrp_stopping_criterion import FirstFeasible
from .request_insertion_feasibility_check import SimpleInsertionRIFC
from .static_routing import StaticPyVrp


def disclosure_time_key(x):
    return x.disclosure_time


class BundleInsertionFeasibilityCheck(ParameterizedClass):
    @abc.abstractmethod
    def __call__(
        self,
        instance: CAHDInstance,
        tours: list[Tour],
        depot: Depot,
        bundle: Bundle,
        max_num_tours: int,
    ) -> bool:
        pass

    def __repr__(self):
        return self.__class__.__name__


class StaticSequentialCheapestInsertionBIFC(BundleInsertionFeasibilityCheck):
    def __init__(
        self,
        sequence_key: callable,
        request_insertion_feasibility_check=SimpleInsertionRIFC(),
        request_insertion=DynamicCheapestInsertion(MinDuration()),
    ):
        self.sequence_key = sequence_key
        self.request_insertion_feasibility_check = request_insertion_feasibility_check
        self.request_insertion = request_insertion
        self._params = {
            "sequence_key": sequence_key,
            "request_insertion_feasibility_check": request_insertion_feasibility_check,
            "request_insertion": request_insertion,
        }

    def __call__(
        self,
        instance: CAHDInstance,
        tours: list[Tour],
        depot: Depot,
        bundle: Bundle,
        max_num_tours: int,
    ) -> bool:
        all_requests = []
        for tour in tours:
            all_requests.extend(tour.requests)
        all_requests.extend(bundle.requests)

        new_tours = []
        for request in sorted(all_requests, key=self.sequence_key):
            if self.request_insertion_feasibility_check(
                instance, new_tours, depot, request, max_num_tours
            ):
                new_tours = self.request_insertion(
                    instance, new_tours, depot, request, max_num_tours
                )
            else:
                return False
        return True


class PyVrpBIFC(BundleInsertionFeasibilityCheck):
    def __init__(self, max_runtime: int = 10, no_improvement: int = 1000):
        """

        :param max_runtime:
        :param no_improvement:
        """
        self._max_runtime = max_runtime
        self._no_improvement = no_improvement
        self._params = {
            "max_runtime": max_runtime,
            "no_improvement": no_improvement,
        }

    def __call__(
        self,
        instance: CAHDInstance,
        tours: list[Tour],
        depot: Depot,
        bundle: Bundle,
        max_num_tours: int,
    ) -> bool:
        stopping_criterion = MultipleCriteria(
            [
                FirstFeasible(),
                MaxRuntime(self._max_runtime),
                NoImprovement(self._no_improvement),
            ]
        )
        all_requests = []
        for tour in tours:
            all_requests.extend(tour.requests)
        all_requests.extend(bundle.requests)
        problem_data = StaticPyVrp._convert_instance_to_pyvrp_data(
            instance, depot, all_requests, len(tours)
        )
        model = pyvrp.Model.from_data(problem_data)
        result = model.solve(stopping_criterion, display=True)
        return result.is_feasible()
