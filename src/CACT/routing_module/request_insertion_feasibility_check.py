import abc

import pyvrp
from pyvrp.stop import MultipleCriteria, MaxRuntime, NoImprovement

from core_module.depot import Depot
from core_module.instance import CAHDInstance
from core_module.request import Request
from core_module.tour import Tour
from utility_module.parameterized_class import ParameterizedClass
from .pyvrp_stopping_criterion import FirstFeasible
from .static_routing import StaticPyVrp


class RequestInsertionFeasibilityCheck(ParameterizedClass):
    def __repr__(self):
        return self.__class__.__name__

    def __call__(
        self,
        instance: CAHDInstance,
        tours: list[Tour],
        depot: Depot,
        request: Request,
        max_num_tours: int,
    ) -> bool:
        # check if opening a new pendulum tour is possible. If not check insertion in existing tours
        if self.check_opening_new_tour(instance, tours, depot, request, max_num_tours):
            return True
        elif tours:
            return self.check_inserting_into_existing_tours(
                instance, tours, depot, request
            )
        else:
            return False

    def check_opening_new_tour(
        self,
        instance: CAHDInstance,
        tours: list[Tour],
        depot: Depot,
        request: Request,
        max_num_tours,
    ) -> bool:
        if len(tours) >= max_num_tours:
            return False
        else:
            tour = Tour(len(tours), depot)
            return tour.insertion_feasibility_check(instance, [1], [request])

    @abc.abstractmethod
    def check_inserting_into_existing_tours(
        self, instance: CAHDInstance, tours: list[Tour], depot: Depot, request: Request
    ) -> bool:
        pass


class PyVrpRIFC(RequestInsertionFeasibilityCheck):
    """
    Just does a complete re-routing, but stops as soon as the first feasible solution was found
    """

    def __init__(self, max_runtime: int = 10, no_improvement: int = 1000):
        """

        :param max_runtime: Maximum runtime stopping criterion in seconds. If no feasible solution has been found after
        max_runtime seconds, the insertion is evaluated as infeasible.
        :param no_improvement: Maximum number of iterations without improvement. If no feasible solution has been found
        after no_improvement iterations, the insertion is evaluated as infeasible.
        """
        self._max_runtime = max_runtime
        self._no_improvement = no_improvement
        self._params = {
            "max_runtime": max_runtime,
            "no_improvement": no_improvement,
        }

    def check_inserting_into_existing_tours(
        self, instance: CAHDInstance, tours: list[Tour], depot: Depot, request: Request
    ) -> bool:
        # stopping criterion must be instantiated for each request, cannot be reused!
        all_requests = []
        for tour in tours:
            all_requests += list(tour.requests)
        all_requests.append(request)
        problem_data = StaticPyVrp._convert_instance_to_pyvrp_data(
            instance, depot, all_requests, len(tours)
        )
        model = pyvrp.Model.from_data(problem_data)
        stopping_criterion = MultipleCriteria(
            [
                FirstFeasible(),
                MaxRuntime(self._max_runtime),
                NoImprovement(self._no_improvement),
            ]
        )
        result = model.solve(stopping_criterion, display=False)
        return result.is_feasible()


class SimpleInsertionRIFC(RequestInsertionFeasibilityCheck):
    """
    Checks all tours and all insertion positions and returns True once the first combination of tour and position is
    feasible.
    In particular, this does not consider the opportunities of re-routing some of the existing requests to make space
    for the new request.
    """

    def check_inserting_into_existing_tours(
        self, instance: CAHDInstance, tours: list[Tour], depot: Depot, request: Request
    ) -> bool:
        for tour in tours:
            for pos in range(1, len(tour)):
                if tour.insertion_feasibility_check(instance, [pos], [request]):
                    return True
        return False
