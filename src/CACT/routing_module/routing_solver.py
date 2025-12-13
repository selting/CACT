from auction_module.bundle_generation.bundle_based.bundle import Bundle
from core_module.depot import Depot
from core_module.instance import CAHDInstance
from core_module.request import Request
from core_module.tour import Tour
from utility_module.parameterized_class import ParameterizedClass
from .bundle_insertion import BundleInsertion
from .bundle_insertion_feasibility_check import BundleInsertionFeasibilityCheck
from .dynamic_routing import DynamicRouting
from .request_insertion_feasibility_check import RequestInsertionFeasibilityCheck
from .static_routing import StaticRouting


class RoutingSolver(ParameterizedClass):
    """
    Basically, this is just a container of different strategies for how to handle routing-based operations.
    """

    def __init__(
        self,
        request_insertion_feasibility_check: RequestInsertionFeasibilityCheck,
        request_insertion: DynamicRouting,
        bundle_insertion_feasibility_check: BundleInsertionFeasibilityCheck,
        bundle_insertion: BundleInsertion,
        static_routing: StaticRouting = None,
    ):
        self._request_insertion_feasibility_check: RequestInsertionFeasibilityCheck = (
            request_insertion_feasibility_check
        )
        self._request_insertion: DynamicRouting = request_insertion
        self._bundle_insertion_feasibility_check: BundleInsertionFeasibilityCheck = (
            bundle_insertion_feasibility_check
        )
        self._bundle_insertion: BundleInsertion = bundle_insertion
        self._static_routing: StaticRouting = static_routing

        self._params = {
            "request_insertion_feasibility_check": request_insertion_feasibility_check,
            "request_insertion": request_insertion,
            "bundle_insertion_feasibility_check": bundle_insertion_feasibility_check,
            "bundle_insertion": bundle_insertion,
            "static_routing": static_routing,
        }
        pass

    def __repr__(self):
        return f"RoutingSolver({self.params})"

    def check_request_insertion_feasibility(
        self,
        instance: CAHDInstance,
        tours: list[Tour],
        depot,
        request: Request,
        max_num_tours,
    ) -> bool:
        return self._request_insertion_feasibility_check(
            instance, tours, depot, request, max_num_tours
        )

    def insert_request(
        self,
        instance: CAHDInstance,
        tours: list[Tour],
        depot: Depot,
        request: Request,
        max_num_tours: int,
    ):
        return self._request_insertion(instance, tours, depot, request, max_num_tours)

    def check_bundle_insertion_feasibility(
        self,
        instance: CAHDInstance,
        tours: list[Tour],
        depot: Depot,
        bundle: Bundle,
        max_num_tours,
    ) -> bool:
        return self._bundle_insertion_feasibility_check(
            instance, tours, depot, bundle, max_num_tours
        )

    def insert_bundle(
        self,
        instance: CAHDInstance,
        tours: list[Tour],
        depot: Depot,
        bundle: Bundle,
        max_num_tours: int,
    ):
        return self._bundle_insertion(instance, tours, depot, bundle, max_num_tours)

    def solve_vrp_statically(
        self,
        instance: CAHDInstance,
        depot: Depot,
        requests: list[Request],
        max_num_tours: int,
    ):
        return self._static_routing(instance, depot, requests, max_num_tours)
