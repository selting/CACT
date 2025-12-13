import abc
from copy import deepcopy

from core_module.depot import Depot
from core_module.instance import CAHDInstance
from core_module.request import Request
from core_module.tour import Tour
from utility_module.parameterized_class import ParameterizedClass
from .insertion_criterion import InsertionCriterion
from .static_routing import StaticPyVrp


class DynamicRouting(ParameterizedClass):
    """
    Dynamic routing strategy that adds one request at a time.
    Caution: never rely on that these insertion strategies check feasibility. Checking feasibility is the task of the
    RequestInsertionFeasibilityCheck.
    """

    @abc.abstractmethod
    def __call__(
        self,
        instance: CAHDInstance,
        tours: list[Tour],
        depot: Depot,
        request: Request,
        max_num_tours: int,
    ) -> list[Tour]:
        """
        :param depot:
        :param instance:
        :param tours:
        :param request:
        :return:
        """
        pass


class DynamicCheapestInsertion(DynamicRouting):
    def __init__(self, insertion_criterion: InsertionCriterion):
        self._insertion_criterion = insertion_criterion
        self._params = {"insertion_criterion": insertion_criterion}

    def __repr__(self):
        return self.__class__.__name__

    def __call__(
        self,
        instance: CAHDInstance,
        tours: list[Tour],
        depot: Depot,
        request: Request,
        max_num_tours: int,
    ) -> list[Tour]:
        new_tours = list(deepcopy(tours))
        # 1. find the best insertion tour and position, also considering opening a new tour
        best_criterion = float("inf")
        best_tour = "infeasible"
        best_pos = "infeasible"

        # check the existing tours
        for tour in new_tours:
            tour_best_delta = float("inf")
            tour_best_pos = None
            for pos in range(1, len(tour)):
                if tour.insertion_feasibility_check(instance, [pos], [request]):
                    criterion = self._insertion_criterion(instance, tour, pos, request)
                    if criterion < tour_best_delta:
                        tour_best_delta = criterion
                        tour_best_pos = pos
            if tour_best_delta < best_criterion:
                best_criterion = tour_best_delta
                best_tour = tour
                best_pos = tour_best_pos

        # 1.2 check opening a new tour
        if len(new_tours) < max_num_tours:
            tour = Tour(len(new_tours), depot)
            pos = 1
            criterion = self._insertion_criterion(instance, tour, pos, request)
            # criterion = criterion * 0.75  # to facilitate opening new tours, only .75 of the pendulum's criterion is accounted for
            if criterion < best_criterion:
                best_criterion = criterion
                best_tour = tour
                best_pos = pos

        # 2. execute the insertion
        if best_tour == "infeasible":
            raise ValueError("No feasible insertion found")
        best_tour.insert_and_update(instance, [best_pos], [request])
        if best_tour not in new_tours:
            # if a new tour was created
            new_tours.append(best_tour)
        return new_tours


class DynamicPyVrp(DynamicRouting):
    """
    Route all requests, including the one that shall be inserted, from scratch and return the new tours.
    """

    def __init__(self, pyvrp_stopping_criterion):
        self._pyvrp_stopping_criterion = pyvrp_stopping_criterion

    @property
    def pyvrp_stopping_criterion(self):
        return deepcopy(self._pyvrp_stopping_criterion)

    def __call__(
        self,
        instance: CAHDInstance,
        tours: list[Tour],
        depot: Depot,
        request: Request,
        max_num_tours: int,
    ) -> list[Tour]:
        all_requests = []
        for tour in tours:
            all_requests += list(tour.requests)
        all_requests.append(request)
        return StaticPyVrp(self.pyvrp_stopping_criterion)(
            instance, depot, all_requests, max_num_tours
        )
