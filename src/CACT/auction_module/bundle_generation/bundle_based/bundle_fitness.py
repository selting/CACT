import warnings
from abc import abstractmethod
from typing import Sequence, Callable

import numpy as np
import pandas as pd

from auction_module.bundle_generation.fitness import Fitness
from auction_module.bundle_generation.bundle_based.bundle import Bundle
from core_module.instance import CAHDInstance
from core_module.solution import CAHDSolution
import datetime as dt


class BundleFitness(Fitness):
    @abstractmethod
    def evaluate(
        self, instance: CAHDInstance, solution: CAHDSolution, bundle: Bundle
    ) -> float:
        pass


class BundleFitnessCarrierAware(Fitness):
    """
    returns one fitness value per carrier
    """

    @abstractmethod
    def evaluate(
        self, instance: CAHDInstance, solution: CAHDSolution, bundle: Bundle
    ) -> Sequence[float]:
        pass


class AggregateCarrierAware(BundleFitness):
    def __init__(
        self, fitness: BundleFitnessCarrierAware, aggregation_function: Callable
    ):
        super().__init__()
        self.fitness = fitness
        self.aggregation_function = aggregation_function

    def evaluate(
        self, instance: CAHDInstance, solution: CAHDSolution, bundle: Bundle
    ) -> float:
        fitness_values = self.fitness.evaluate(instance, solution, bundle)
        return self.aggregation_function(fitness_values)


class TrueBids(
    BundleFitnessCarrierAware
):  # Carrier-aware fitness, i.e. returns num_carriers fitness values
    def __init__(self, bidding_behavior):
        super().__init__()
        self.bidding_behavior = bidding_behavior

    def evaluate(
        self, instance: CAHDInstance, solution: CAHDSolution, bundle: Bundle
    ) -> float:
        bundle_bidding_jobs = pd.DataFrame(
            data=np.full((1, instance.num_carriers), True),
            index=[bundle],
            columns=solution.carriers,
        )
        bids_matrix = self.bidding_behavior.execute_bidding(
            instance, solution, bundle_bidding_jobs, disable_tqdm=True
        )
        bids_matrix = bids_matrix.applymap(lambda x: x.total_seconds())
        # smaller z value is better
        return (-bids_matrix).squeeze().to_list()


class RuetherRieck2021(BundleFitnessCarrierAware):
    """
    Based on "Rüther, C., & Rieck, J. (2022). Bundle selection approaches for collaborative practical-oriented Pickup
    and Delivery Problems. EURO Journal on Transportation and Logistics, 11, 100087."
    """

    def evaluate(
        self, instance: CAHDInstance, solution: CAHDSolution, bundle: Bundle
    ) -> Sequence[float]:
        sum_revenues = sum(
            self.request_revenue(instance, request) for request in bundle.requests
        )
        num_required_vehicles = 1  # simplified
        density = bundle.density  # modified, they have PDP requests
        tour_duration = bundle.tour_sum_travel_duration
        if tour_duration >= dt.timedelta.max.total_seconds():
            warnings.warn(f"No tour feasible for bundle {bundle}")
        # maybe increase the num_required_vehicles if the tour duration is too long?

        alpha, beta, gamma, mu, nu, tau = 0.15, 1.8, 0.2, 0, 0, 1.5  # see paper
        fitness = []
        for carrier in solution.carriers:
            depot_distance = bundle.sum_duration_to_carrier_depot(carrier.depot_vertex)
            f = (
                sum_revenues**alpha
                * bundle.cardinality**beta
                * num_required_vehicles**gamma
                * density**mu
            ) / (tour_duration**nu * depot_distance**tau)
            fitness.append(f)
        return fitness

    def request_revenue(self, instance: CAHDInstance, request: int) -> float:
        original_carrier = instance.request_to_carrier_assignment[request]
        request_vertex = instance.vertex_from_request(request)
        shuttle_tour_duration = instance.travel_duration(
            [original_carrier, request_vertex], [request_vertex, original_carrier]
        )
        cost_factor = 1.0  # see paper
        roh_var = 1.05  # see paper
        revenue = roh_var * cost_factor * shuttle_tour_duration.total_seconds()
        return revenue


class CentroidSSE(BundleFitness):
    def evaluate(
        self, instance: CAHDInstance, solution: CAHDSolution, bundle: Bundle
    ) -> float:
        sse = bundle.sum_of_squared_centroid_distance
        # smaller is better, so we return the negative value
        return -sse
