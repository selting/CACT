from statistics import mean
from typing import Sequence, Set, Callable

import numpy as np

from auction_module.bundle_generation.bundle_based.bundle import Bundle
from auction_module.bundle_generation.partition_based.partition import Partition
from auction_module.bundling_and_bidding.fitness_functions.bundle_fitness import BundleFitnessFunction
from auction_module.bundling_and_bidding.fitness_functions.fitness_functions import FitnessFunction
from auction_module.bundling_and_bidding.type_defs import QueriesType, ResponsesType
from core_module.instance import CAHDInstance
from core_module.request import Request


class FitnessPartitionFeature(FitnessFunction):
    def __init__(self, partition_feature, higher_is_better):
        super().__init__()
        self.search_space = 'partition'
        self.partition_feature = partition_feature
        self.higher_is_better = higher_is_better
        self.optimization_direction = 1 if higher_is_better else -1

    def __repr__(self):
        prefix = '' if self.higher_is_better else '-'
        return prefix + self.partition_feature

    def __call__(self, instance, partitions: Sequence[Partition], **kwargs):
        fitness = []
        for i, binary in enumerate(partitions):
            partition = Partition.from_binary(instance, self.auction_request_pool, binary)
            value = partition.get_feature(self.partition_feature)
            fitness.append(self.optimization_direction * value)
        return fitness


class FitnessPartitionGanstererHartl(FitnessFunction):
    def __init__(self):
        super().__init__()
        self.search_space = 'partition'
        pass

    def __call__(self, instance, partitions: Sequence[Partition], **kwargs):
        """
        This fitness function is based on the paper "Gansterer, M., & Hartl, R. F. (2018). Centralized bundle
        generation in auction-based collaborative transportation. OR spectrum quantitative approaches in management,
        40, 613–635."
        :param instance:
        :param **kwargs:
        """
        fitness = []
        for i, binary in enumerate(partitions):
            partition = Partition.from_binary(self.instance, self.auction_request_pool, binary)
            partition.normalize(True)
            if partition.k == 1:
                fitness.append(0)
                continue
            non_empty_bundles = [b for b in partition.bundles if any(b.bitstring)]
            isolations = [self.isolation(b, set(non_empty_bundles) - {b}) for b in non_empty_bundles]
            # density is not computed exactly like in the paper, but they have a PDP
            densities = [bundle.density for bundle in non_empty_bundles]
            tour_length = [bundle.tour_sum_travel_duration for bundle in non_empty_bundles]
            if any(np.isnan(x) for x in tour_length):
                gh_proxy = 0
            else:
                gh_proxy = (min(isolations) * min(densities)) / (max(tour_length) * len(non_empty_bundles))
            fitness.append(gh_proxy)
        return fitness

    def isolation(self, bundle: Bundle, other_bundles: Set[Bundle]):
        """This parameter approximates the separation from other bundles."""
        separations = []
        for other_bundle in other_bundles:
            separations.append(self.separation(bundle, other_bundle))
        return min(separations)

    def separation(self, bundle: Bundle, other_bundle: Bundle):
        """
        This parameter approximates the separation from another bundle by computing the distance between the centroids
        divided by the maximum radius of the two bundles.
        :param bundle:
        :param other_bundle:
        :return:
        """
        centroid_distance = np.linalg.norm(np.array(bundle.spatial_centroid) - np.array(other_bundle.spatial_centroid))
        if bundle.cardinality == 1 and other_bundle.cardinality == 1:
            return centroid_distance
        radius = mean(bundle.centroid_euclidean_distances.values())
        other_radius = mean(other_bundle.centroid_euclidean_distances.values())
        return centroid_distance / max(radius, other_radius)


class FitnessPartitionAggregateBundleFitness(FitnessFunction):
    """
    fitness of the partition is calculated as an aggregation of the bidder-specific fitness of the bundles in the
     partition. The aggregation function is given by the aggr parameter and should accept an m x n matrix, where m is
     the number of bidders and n is the number of bundles in the partition. It should return a scalar.
    """

    def __init__(self, bundle_fitness: BundleFitnessFunction, aggr: Callable, higher_is_better: bool = True):
        super().__init__()
        self.search_space = 'partition'
        self._bundle_fitness: BundleFitnessFunction = bundle_fitness
        self._aggr = aggr
        self._higher_is_better = higher_is_better
        self._optimization_direction = 1 if higher_is_better else -1

        self._params = {
            'bundle_fitness': self._bundle_fitness,
            'aggr': self._aggr,
            'higher_is_better': self._higher_is_better,
        }

    def __repr__(self):
        prefix = '' if self._higher_is_better else '-'
        aggr_name = self._aggr.__name__ if callable(self._aggr) else self._aggr
        return f'{prefix}{aggr_name}({str(self._bundle_fitness)})'

    def __call__(self, instance: CAHDInstance, partitions: Sequence[Partition], **kwargs):
        partitions_fitness = []
        for partition in partitions:
            partition.normalize(True)
            non_empty_bundles = [b for b in partition.bundles if any(b.bitstring)]
            fitness = []
            for bidder_idx in range(instance.num_carriers):
                bidder_fitness = []
                for bundle in non_empty_bundles:
                    bidder_fitness.append(self._bundle_fitness(instance, [bundle], bidder_idx=bidder_idx))
                fitness.append(bidder_fitness)
            fitness_arr = np.array(fitness)  # .reshape(len(binary), -1)
            value = self._aggr(fitness_arr)
            partitions_fitness.append(self._optimization_direction * value)
        return partitions_fitness

    def fit(self, instance: QueriesType, auction_request_pool: tuple[Request], queries: QueriesType,
            responses: ResponsesType):
        return self._bundle_fitness.fit(queries, auction_request_pool, responses)
