from __future__ import annotations

from functools import cached_property
from typing import Sequence, Callable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn import metrics

from auction_module.bundle_generation.bundle_based.bundle import Bundle
from auction_module.bundle_generation.partition_based.random_max_k_partition import (
    random_max_k_partition,
)
from core_module import instance as it
from core_module.request import Request
from utility_module import utils as ut


class Partition:
    def __init__(
        self,
        instance: it.CAHDInstance,
        auction_request_pool: tuple[Request],
        partition_labels: Sequence[int],
    ):
        """

        :param instance:
        :param auction_request_pool:
        :param partition_labels:
        """
        self.instance = instance  # TODO ensure that this is reference and not a copy?
        self.auction_request_pool = tuple(auction_request_pool)
        """tuple of REQUESTS indices of requests in the auction request pool"""
        self.labels = tuple(partition_labels)
        """Restricted Growth String (RGS) representation of the partition"""

        self.n = len(auction_request_pool)
        """number of elements in the set"""

        self.k = len(set(partition_labels))
        """number of blocks into which the set is partitioned"""

        if self._check_is_normalized():
            self.is_normalized = True
        else:
            self.is_normalized = False

    def __repr__(self):
        return f"Partition (n={self.n}, k={self.k}): {self.labels}"

    def __eq__(self, other):
        assert self.is_normalized and other.is_normalized, (
            "cannot compare partitions that are not normalized"
        )
        return (
            self.instance is other.instance
            and self.auction_request_pool == other.auction_request_pool
            and self.labels == other.labels
        )

    def __hash__(self):
        return hash((self.instance, self.auction_request_pool, self.labels))

    @classmethod
    def random_max_k(
        cls, instance: it.CAHDInstance, auction_request_pool: tuple[Request]
    ) -> Partition:
        part = random_max_k_partition(auction_request_pool, instance.num_carriers)
        idx_labels = {
            x: None for x in auction_request_pool
        }  # copy and then replace by idx label
        for i, block in enumerate(part):
            for j, request in enumerate(block):
                idx_labels[request] = i
        idx_labels = [idx_labels[request] for request in auction_request_pool]
        return Partition(instance, auction_request_pool, idx_labels)

    @classmethod
    def from_assignment(cls, assignment):
        # assignment values sorted by key
        partition_labels = [assignment[i] for i in sorted(assignment.keys())]
        return cls(assignment.instance, assignment.requests(), partition_labels)

    @cached_property
    def bundles(
        self,
    ) -> tuple[Bundle, ...]:  # TODO correct terminology would be "blocks"
        return tuple(
            Bundle(self.auction_request_pool, b)
            for b in ut.indices_to_nested_lists(self.labels, self.auction_request_pool)
        )

    def normalize(self, in_place=False):
        """normalize the partition, i.e. transform it into a valid RGS"""
        if self.is_normalized:
            if in_place:
                pass
            else:
                return self

        normalized = list(self.labels[:])
        mapping_idx = 0

        # +1 because the CREATE mutation may exceed the valid num_bundles temporarily
        mapping: list[int] = [-1] * (len(normalized) + 1)
        mapping[normalized[0]] = mapping_idx
        normalized[0] = mapping_idx

        for i in range(1, len(normalized)):
            if mapping[normalized[i]] == -1:
                mapping_idx += 1
                mapping[normalized[i]] = mapping_idx
                normalized[i] = mapping_idx
            else:
                normalized[i] = mapping[normalized[i]]
        if in_place:
            self.labels = tuple(normalized)
            self.is_normalized = True
            try:
                del self.bundles  # if the cached_property has been computed, delete it
            except AttributeError:
                pass
        else:
            return Partition(self.instance, self.auction_request_pool, normalized)

    def _check_is_normalized(self):
        """check if the sequence of labels is normalized, i.e. if it is a valid RGS"""
        if self.labels[0] != 0:
            return False
        for i in range(len(self.labels) - 1):
            if self.labels[i + 1] > 1 + max(self.labels[: i + 1]):
                return False
        return True

    def request_pool_coordinates(self):
        request_pool_coords = []
        for request in self.auction_request_pool:
            request_pool_coords.append((request.x, request.y))
        return request_pool_coords

    @cached_property
    def bundle_centroids_euclidean_distance_matrix(self):
        """matrix of pairwise euclidean distance between bundle centroids"""
        centroids = [b.spatial_centroid for b in self.bundles]

        matrix = [[0] * self.k for _ in range(self.k)]
        for i in range(self.k):
            for j in range(i + 1, self.k):
                centroid_i = self.bundles[i].spatial_centroid
                centroid_j = self.bundles[j].spatial_centroid
                matrix[i][j] = matrix[j][i] = ut.euclidean_distance(
                    centroid_i, centroid_j
                )
        return matrix

    @cached_property
    def sum_centroid_separation(self):  # FIXME better name!
        """
        sum of squared distances of centroids to the overall centroid of all requests in the pool (Tan et al. 2019,
        Ch. 5.5, S. 359, "between group sum of squares (SSB), which is the sum of the squared distance of a cluster
        centroid, ci, to the overall mean, c, of all the data points.")
        """
        request_pool_coords = self.request_pool_coordinates()
        # compute the mean of all x,y coords in request_pool_coords
        mean_x = sum([x for x, y in request_pool_coords]) / len(request_pool_coords)
        mean_y = sum([y for x, y in request_pool_coords]) / len(request_pool_coords)

        total_ssb = 0
        for bundle in self.bundles:
            centroid = bundle.spatial_centroid
            total_ssb += (self.n / bundle.cardinality) * ut.euclidean_distance(
                *centroid, mean_x, mean_y
            ) ** 2

        return total_ssb

    @cached_property
    def silhouette_score(self):
        """
        Advantages:
        The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. Scores around zero
        indicate overlapping clusters.
        The score is higher when clusters are dense and well separated, which relates to a standard concept of a
        cluster.
        Drawbacks:
        The Silhouette Coefficient is generally higher for convex clusters than other concepts of clusters, such as
        density based clusters like those obtained through DBSCAN.
        :return:
        """
        if self.k < 2:
            return 0  # return 0 if there is only one cluster because the silhouette score is not defined
        return metrics.silhouette_score(
            self.request_pool_coordinates(), self.labels, metric="euclidean"
        )

    def global_features(self):
        global_features = dict(
            k=self.k,
            n=self.n,
        )
        global_features["sum_centroid_separation"] = self.sum_centroid_separation
        global_features["silhouette_coefficient"] = self.silhouette_score
        return global_features

    def get_feature(self, feature: str):
        return self.__getattribute__(feature)

    def bundle_features(self, max_k: int):
        """

        :param max_k: for consistent representation of partitions as samples for a neural network, we need to pad the
        bundle features with zeros up to a maximum number of bundles (i.e., max_k)
        :return:
        """
        bundle_features = []
        for bundle in self.bundles:
            bf = bundle.global_features()
            for carrier in range(self.instance.num_carriers):
                bf.update(bundle.carrier_specific_features(carrier))
            bundle_features.append(bf)
        if len(bundle_features) < max_k:
            for _ in range(max_k - len(bundle_features)):
                bf = {k: None for k in bundle_features[0].keys()}
                bundle_features.append(bf)
        return bundle_features

    def carrier_specific_features(self):
        pass

    def aggr_bundle_feature(self, aggr: Callable, bundle_feature: str):
        return aggr([b.get_feature(bundle_feature) for b in self.bundles])

    def all_aggr_bundle_features(
        self, aggregations: Sequence[Callable] = ("sum", "min", "max", "mean", "std")
    ):
        data = []
        for bundle in self.bundles:
            x = dict(bundle=bundle)
            x.update(bundle.global_features())
            for carrier in range(self.instance.num_carriers):
                x.update(bundle.carrier_specific_features(carrier))
            data.append(x)
        df = pd.DataFrame(data).set_index("bundle")

        aggr_df = df.agg(func=aggregations, axis=0)
        return aggr_df

    def as_record(self):
        # record = dict(**instance.meta, **solution.solver_config)
        # record['time_window_length'] = record['time_window_length'].total_seconds()
        # record['time_window_length_minutes'] = record['time_window_length'] / 60
        # record['time_window_length_hours'] = record['time_window_length'] / 60 ** 2
        # record['git_hash'] = io.get_git_hash()
        record = dict()

        record["request_pool"] = ";".join(str(x) for x in self.auction_request_pool)
        record["partition_labels"] = ";".join(str(x) for x in self.labels)

        # global features
        global_features = self.global_features()
        record.update({f"global_feature_{k}": v for k, v in global_features.items()})
        for b_idx, bf in enumerate(
            self.bundle_features(max_k=self.instance.num_carriers)
        ):
            record.update({f"bundle_{b_idx}_{k}": v for k, v in bf.items()})
        # carrier specific features???

        return record

    # def overlap(self, other_partition: Partition):
    #     """
    #     Compute the overlap of two partitions. The overlap is defined as the number of requests that are in the same
    #     bundle in both partitions. Example:
    #     partition 1: {{0, 1}, {2}, {3, 4}}
    #     partition 2: {{0, 1, 2}, {3, 4}}
    #     overlap: 2/5 = 0.4 (requests 3 and 4 are in the same bundle in both partitions)
    #     """
    #     assert self.instance is other_partition.instance, 'cannot compare partitions of different instances'
    #     assert self.auction_request_pool == other_partition.auction_request_pool, \
    #         'cannot compare partitions with different request pools'
    #     overlap = 0
    #     for bundle in self.bundles:
    #         for other_bundle in other_partition.bundles:
    #             if bundle == other_bundle:
    #                 overlap += bundle.cardinality
    #     return overlap / self.n

    def misclassification_error(self, other: Partition):
        """
        computes the misclassification error between two partitions.
        It is also known as "transfer distance", "partition distance", or "maximum matching distance".
        See Hennig, C., Meilǎ, M., Murtagh, F., & Rocci, R. (2015). Handbook of Cluster Analysis: Chapman and Hall/CRC.

        :param other:
        :return:
        """
        assert self.instance is other.instance, (
            "cannot compare partitions of different instances"
        )
        assert self.auction_request_pool == other.auction_request_pool, (
            "cannot compare partitions with different request pools"
        )
        assert self.k <= self.instance.num_carriers

        # we must ensure that K <= K'
        if self.k > other.k:
            return other.misclassification_error(self)

        # NOTE A) solving with gurobi and the model found in:
        #  Hennig, C., Meilǎ, M., Murtagh, F., & Rocci, R. (2015). Handbook of Cluster Analysis: Chapman and Hall/CRC.
        # with gp.Env(empty=True) as env:
        #     env.setParam('OutputFlag', 0)
        #     env.start()
        #     lp: gp.Model
        #     with gp.Model(env=env) as lp:
        #         coefficients = gp.tupledict()
        #         for k_self, bundle_self in enumerate(self.bundles):
        #             for k_other, bundle_other in enumerate(other.bundles):
        #                 intersection = set(bundle_self.requests).intersection(bundle_other.requests)
        #                 probability = len(intersection) / self.n
        #                 coefficients[k_self, k_other] = probability
        #         x = lp.addVars(coefficients.keys(), vtype=GRB.BINARY, name='x')
        #         objective = 1 - x.prod(coefficients)
        #         lp.setObjective(objective, GRB.MINIMIZE)
        #         for k_other in range(other.k):
        #             lp.addConstr(x.sum('*', k_other) == 1, name=f'other_{k_other}')
        #         for k_self in range(self.k):
        #             lp.addConstr(x.sum(k_self, '*') == 1, name=f'self_{k_self}')
        #         lp.optimize()
        #         misclassification_error = lp.objVal

        # NOTE B) solve using scipy linear_sum_assignment
        coefficients_dim0 = []
        for k_self, bundle_self in enumerate(self.bundles):
            coefficients_dim1 = []
            for k_other, bundle_other in enumerate(other.bundles):
                intersection = set(bundle_self.requests).intersection(
                    bundle_other.requests
                )
                probability = len(intersection) / self.n
                coefficients_dim1.append(probability)
            coefficients_dim0.append(coefficients_dim1)
        coefficients_dim0 = np.array(coefficients_dim0)
        row_ind, col_ind = linear_sum_assignment(
            -coefficients_dim0
        )  # switch the sign because scipy minimizes
        misclassification_error = 1 - coefficients_dim0[row_ind, col_ind].sum()
        return misclassification_error

    @classmethod
    def from_binary(cls, instance, auction_request_pool: tuple[Request], binary):
        idx_labels = {
            x: None for x in auction_request_pool
        }  # copy and then replace by idx label
        for i, block in enumerate(binary):
            request_bundle = [
                auction_request_pool[j] for j, x in enumerate(block) if x == 1
            ]
            for j, request in enumerate(request_bundle):
                idx_labels[request] = i
        idx_labels = [idx_labels[request] for request in auction_request_pool]
        return Partition(instance, auction_request_pool, idx_labels)


def random_partition(n, k):
    """
    create a random k-partition of a set of n elements.
    Note: it is possible that some of the k subsets are the empty set.
    :param n:
    :param k:
    :return:
    """
    from utility_module.random import rng

    s = [[0 for i in range(n)] for j in range(k)]
    for i in range(n):
        j = rng.integers(0, len(s))
        s[j][i] = 1
    return set(tuple(j) for j in s)
