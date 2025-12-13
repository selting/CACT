from abc import abstractmethod
from statistics import mean
from typing import Callable, Sequence

import numpy as np

from auction_module.bundle_generation.bundle_based.bundle import Bundle
from auction_module.bundle_generation.fitness import Fitness
from auction_module.bundle_generation.partition_based.partition import Partition
from core_module.instance import CAHDInstance
from core_module.solution import CAHDSolution


class PartitionFitness(Fitness):
    """
    Abstract class for fitness functions for partitions.
    In general, it holds that the higher the fitness value, the better the individual.
    """

    @abstractmethod
    def evaluate(
        self, instance: CAHDInstance, solution: CAHDSolution, partition: Partition
    ) -> float:
        pass


class AggregatePartitionFitness(PartitionFitness):
    def __init__(self, aggr: Callable):
        super().__init__()
        self.aggr = aggr
        self.name = aggr.__name__.capitalize() + "_" + self.__class__.__name__

    @abstractmethod
    def evaluate(
        self, instance: CAHDInstance, solution: CAHDSolution, partition: Partition
    ) -> float:
        pass


class CentroidSSE(AggregatePartitionFitness):
    def evaluate(
        self, instance: CAHDInstance, solution: CAHDSolution, partition: Partition
    ) -> float:
        sse = self.aggr(
            bundle.sum_of_squared_centroid_distance for bundle in partition.bundles
        )
        # smaller sse is better
        return -sse


class GanstererHartl2018(PartitionFitness):
    def evaluate(
        self, instance: CAHDInstance, solution: CAHDSolution, partition: Partition
    ) -> float:
        """
        This fitness function is based on the paper "Gansterer, M., & Hartl, R. F. (2018). Centralized bundle
        generation in auction-based collaborative transportation. OR spectrum quantitative approaches in management,
        40, 613–635."
        """
        isolations = [
            self.isolation(b, set(partition.bundles) - {b}) for b in partition.bundles
        ]
        densities = [
            bundle.density for bundle in partition.bundles
        ]  # not exactly as in the paper, but they have a PDP
        tour_length = [bundle.tour_sum_travel_duration for bundle in partition.bundles]
        gh_proxy = (min(isolations) * min(densities)) / (
            max(tour_length) * len(partition.bundles)
        )
        return gh_proxy

    def isolation(self, bundle: Bundle, other_bundles: Sequence[Bundle]):
        """This parameter approximates the separation from other bundles."""
        separations = []
        for other_bundle in other_bundles:
            separations.append(self.separation(bundle, other_bundle))
        return min(separations)

    def separation(self, bundle: Bundle, other_bundle: Bundle):
        centroid_distance = np.linalg.norm(
            np.array(bundle.spatial_centroid) - np.array(other_bundle.spatial_centroid)
        )
        radius = mean(bundle.centroid_euclidean_distances.values())
        other_radius = mean(other_bundle.centroid_euclidean_distances.values())
        return centroid_distance / max(radius, other_radius)


class SpatialCohesion(PartitionFitness):
    """
    The quality of a partition is defined as the (negative) spatial cohesion of requests,aggregated by the
    given aggr function
    """

    def __init__(self, aggr: Callable):
        super().__init__()
        self.aggr = aggr
        self.name = aggr.__name__.capitalize() + self.__class__.__name__

    def evaluate(
        self, instance: CAHDInstance, solution: CAHDSolution, partition: Partition
    ) -> float:
        spatial_cohesion = self.aggr(
            b.mean_pairwise_duration for b in partition.bundles
        )
        # smaller cohesion is better
        return -spatial_cohesion
