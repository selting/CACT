import abc

from auction_module.bundle_generation.bundle_based.bundle import Bundle
from core_module.depot import Depot
from core_module.instance import CAHDInstance
from core_module.tour import Tour
from utility_module.parameterized_class import ParameterizedClass
from .insertion_criterion import InsertionCriterion
from .static_routing import StaticSequentialCheapestInsertion, StaticPyVrp


class BundleInsertion(ParameterizedClass):
    @abc.abstractmethod
    def __call__(
        self,
        instance: CAHDInstance,
        tours: list[Tour],
        depot: Depot,
        bundle: Bundle,
        max_num_tours: int,
    ) -> list[Tour]:
        pass

    def __repr__(self):
        return self.__class__.__name__


class StaticSequentialCheapestInsertionBundleInsertion(BundleInsertion):
    """
    Re-routes from scratch by assembling all requests (those that are already routed and the bundle requests) and
    then calling the sequential cheapest insertion static routing solver (inserts one request at a time, in order of
    the sequence_key
    """

    def __init__(
        self,
        sequence_key: callable,
        insertion_criterion: InsertionCriterion,
    ):
        self._static_routing = StaticSequentialCheapestInsertion(
            sequence_key, insertion_criterion
        )
        self._params = {"insertion_criterion": insertion_criterion}

    def __call__(
        self,
        instance: CAHDInstance,
        tours: list[Tour],
        depot: Depot,
        bundle: Bundle,
        max_num_tours: int,
    ) -> list[Tour]:
        all_requests = []
        for tour in tours:
            all_requests.extend(tour.requests)
        all_requests.extend(bundle.requests)
        new_tours = self._static_routing(instance, depot, all_requests, max_num_tours)
        return new_tours


class CheapestCheapestInsertionBundleInsertion(BundleInsertion):
    """
    Inserts requests one by one, through repeatedly finding the best combination of request, tour and insertion position

    """

    pass


class PyVrpBundleInsertion(BundleInsertion):
    """
    Returns a new set of tours that includes the bundle by simpyl re-routing from scratch using PyVrp
    """

    def __init__(self, stopping_criterion):
        self.stopping_criterion = stopping_criterion
        self._params = {"stopping_criterion": stopping_criterion}

    def __call__(
        self,
        instance: CAHDInstance,
        tours: list[Tour],
        depot: Depot,
        bundle: Bundle,
        max_num_tours: int,
    ) -> list[Tour]:
        all_requests = []
        for tour in tours:
            all_requests += list(tour.requests)
        all_requests.extend(bundle.requests)
        return StaticPyVrp(self.stopping_criterion)(
            instance, depot, all_requests, max_num_tours
        )
