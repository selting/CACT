import abc
import warnings
from copy import deepcopy
from typing import Callable

import pyvrp
from tqdm import tqdm

from core_module.depot import Depot
from core_module.instance import CAHDInstance
from core_module.request import Request
from core_module.tour import Tour
from utility_module.datetime_handling import total_seconds_vectorized
from utility_module.parameterized_class import ParameterizedClass
from utility_module.utils import ACCEPTANCE_START_TIME, EXECUTION_START_TIME, END_TIME
from .insertion_criterion import InsertionCriterion


class StaticRouting(ParameterizedClass):
    @abc.abstractmethod
    def __call__(
        self,
        instance: CAHDInstance,
        depot: Depot,
        requests: list[Request],
        max_num_tours: int,
    ) -> list[Tour]:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}"


class StaticSequentialInsertion(StaticRouting):
    """
    Insert requests in the order in which they are supplied. Once a route is full, the next one is opened.
    """

    def __call__(
        self,
        instance: CAHDInstance,
        depot: Depot,
        requests: list[Request],
        max_num_tours: int,
    ) -> list[Tour]:
        tours = [Tour(i, depot) for i in range(max_num_tours)]
        for request in requests:
            for tour in tours:
                pos = len(tour) - 1
                if tour.insertion_feasibility_check(instance, [pos], [request]):
                    tour.insert_and_update(instance, [pos], [request])
                    break
        return tours


class StaticSequentialCheapestInsertion(StaticRouting):
    def __init__(self, sequence_key: Callable, insertion_criterion: InsertionCriterion):
        self._sequence_key = sequence_key
        self._insertion_criterion = insertion_criterion
        self._params = {
            "sequence_key": sequence_key,
            "insertion_criterion": insertion_criterion,
        }

    def __call__(
        self,
        instance: CAHDInstance,
        depot: Depot,
        requests: list[Request],
        max_num_tours: int,
    ) -> list[Tour]:
        tours = [Tour(vehicle_idx, depot) for vehicle_idx in range(max_num_tours)]
        for request in sorted(requests, key=self._sequence_key):
            tours = self._request_insertion(
                instance, tours, depot, request, max_num_tours
            )
        return tours

    def _request_insertion(
        self,
        instance: CAHDInstance,
        tours: list[Tour],
        depot: Depot,
        request: Request,
        max_num_tours: int,
    ) -> list[Tour]:
        # duplicate, because of circular import issues
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


class StaticCheapestCheapestInsertion(StaticRouting):
    def __init__(self, insertion_criterion: InsertionCriterion):
        self.insertion_criterion = insertion_criterion
        self._params = {"insertion_criterion": insertion_criterion}

    def __call__(
        self,
        instance: CAHDInstance,
        depot: Depot,
        requests: list[Request],
        max_num_tours: int,
    ) -> list[Tour]:
        tours = [Tour(vehicle_idx, depot) for vehicle_idx in range(max_num_tours)]
        unrouted_requests = [r for r in requests]
        routed_requests = []
        pbar = tqdm(total=len(requests))
        while unrouted_requests:
            best_delta = float("inf")
            best_request = None
            best_pos = None
            for request in unrouted_requests:
                request_best_tour = None
                request_best_pos = None
                request_best_delta = float("inf")
                for tour in tours:
                    tour_best_delta = float("inf")
                    tour_best_pos = None
                    for pos in range(1, len(tour)):
                        # TODO: decouple the max_distance and duration from the instance?
                        if tour.insertion_feasibility_check(instance, [pos], [request]):
                            delta = self.insertion_criterion(
                                instance, tour, pos, request
                            )
                            if delta < tour_best_delta:
                                tour_best_delta = delta
                                tour_best_pos = pos
                    if tour_best_delta < request_best_delta:
                        request_best_delta = tour_best_delta
                        request_best_pos = tour_best_pos
                        request_best_tour = tour
                if request_best_delta < best_delta:
                    best_delta = request_best_delta
                    best_request = request
                    best_pos = request_best_pos
                    best_tour = request_best_tour

            best_tour.insert_and_update(instance, [best_pos], [best_request])
            routed_requests.append(best_request)
            unrouted_requests.remove(best_request)
            pbar.update(1)
        return tours


class StaticPyVrp(StaticRouting):
    def __init__(self, pyvrp_stopping_criterion):
        self._pyvrp_stopping_criterion = pyvrp_stopping_criterion
        self._params = {"pyvrp_stopping_criterion": pyvrp_stopping_criterion}

    @property
    def pyvrp_stopping_criterion(self):
        return deepcopy(self._pyvrp_stopping_criterion)

    def __call__(
        self,
        instance: CAHDInstance,
        depot: Depot,
        requests: list[Request],
        max_num_tours: int,
    ) -> list[Tour]:
        pyvrp_problem_data = self._convert_instance_to_pyvrp_data(
            instance, depot, requests, max_num_tours
        )
        pyvrp_model = pyvrp.Model.from_data(pyvrp_problem_data)
        pyvrp_stopping_criterion = self.pyvrp_stopping_criterion
        pyvrp_result = pyvrp_model.solve(pyvrp_stopping_criterion, display=False)
        if pyvrp_result.is_feasible():
            return self._convert_pyvrp_result_to_tours(
                instance, depot, requests, pyvrp_result
            )
        else:
            warnings.warn(
                "PyVRP did not find a feasible solution. You may want to double check the stopping criteria."
            )
            return "infeasible"

    @staticmethod
    def _convert_instance_to_pyvrp_data(
        instance: CAHDInstance,
        depot: Depot,
        requests: list[Request],
        max_num_vehicles: int,
    ) -> pyvrp.ProblemData:
        if "vienna" in instance.meta["t"]:
            clients = []
            sorted_requests = sorted(requests, key=lambda x: x.uid)
            coords_scaling_factor = 1e14
            for r in sorted_requests:
                c = pyvrp.Client(
                    x=round(r.x * coords_scaling_factor),
                    y=round(r.y * coords_scaling_factor),
                    delivery=r.load,
                    pickup=0,
                    service_duration=r.service_duration.total_seconds(),
                    tw_early=(r.tw_open - ACCEPTANCE_START_TIME).total_seconds(),
                    tw_late=(r.tw_close - ACCEPTANCE_START_TIME).total_seconds(),
                    release_time=(
                        r.disclosure_time - ACCEPTANCE_START_TIME
                    ).total_seconds(),
                    name="uid=" + str(r.uid),
                )
                clients.append(c)
            depots = [
                pyvrp.Depot(
                    x=round(depot.x * coords_scaling_factor),
                    y=round(depot.y * coords_scaling_factor),
                    name=depot.label,
                )
            ]
            vehicle_types = [
                pyvrp.VehicleType(
                    num_available=max_num_vehicles,
                    capacity=round(instance.max_vehicle_load),
                    start_depot=0,  # there is only one depot here
                    end_depot=0,  # there is only one depot here
                    tw_early=(
                        EXECUTION_START_TIME - ACCEPTANCE_START_TIME
                    ).total_seconds(),
                    tw_late=(END_TIME - ACCEPTANCE_START_TIME).total_seconds(),
                    max_duration=round(instance.max_tour_duration.total_seconds()),
                    max_distance=round(instance.max_tour_distance),
                    name="VehicleType 0",
                    unit_distance_cost=0,
                    unit_duration_cost=1,
                )
            ]
            included_vertices = [depot.uid] + [r.uid for r in sorted_requests]
            distance_matrices = [
                instance._travel_distance_matrix[included_vertices][
                    :, included_vertices
                ]
            ]

            dur_matr = instance._travel_duration_matrix[included_vertices][
                :, included_vertices
            ]
            duration_matrices = [total_seconds_vectorized(dur_matr).astype(int)]

            pdata = pyvrp.ProblemData(
                clients=clients,
                depots=depots,
                vehicle_types=vehicle_types,
                distance_matrices=distance_matrices,
                duration_matrices=duration_matrices,
            )
        else:
            raise ValueError("Only vienna instances are supported for now")
        return pdata

    @staticmethod
    def _convert_pyvrp_result_to_tours(
        instance: CAHDInstance,
        depot: Depot,
        included_requests: list[Request],
        result: pyvrp.Result,
    ) -> list[Tour]:
        sorted_requests = sorted(included_requests, key=lambda x: x.uid)
        solution = result.best  # best found solution by pyvrp
        tours = []
        for i, pyvrp_route in enumerate(solution.routes()):
            tour = Tour(i, depot)
            for pyvrp_index in pyvrp_route.visits():
                request = sorted_requests[pyvrp_index - 1]
                insertion = len(tour) - 1
                tour.insert_and_update(instance, [insertion], [request])
            tours.append(tour)
        return tours
