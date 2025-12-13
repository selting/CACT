import datetime as dt
import math
import random
import sqlite3
from functools import cached_property
from statistics import mean, stdev
from typing import Sequence, Optional

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull

from core_module.depot import Depot
from core_module.instance import CAHDInstance
from core_module.request import Request
from core_module.tour import Tour
from tw_management_module.time_window import TimeWindow
from utility_module import utils as ut
from utility_module.geometry import compute_angle_degrees
from utility_module.utils import EXECUTION_START_TIME


class Bundle:
    def __init__(self, all_items: Sequence[Request], bundle_items: Sequence[Request]):
        assert len(bundle_items) == len(set(bundle_items)), (
            f"bundle cannot contain duplicates"
        )
        assert len(all_items) == len(set(all_items)), f"all_items contain duplicates"
        assert all(
            all_items[i] <= all_items[i + 1] for i in range(len(all_items) - 1)
        ), "all_items must be sorted"  # NOTE: WHY do they have to be sorted?
        assert all(x in all_items for x in bundle_items), (
            f"subset {bundle_items} contains values that are not present in set_ {all_items}"
        )

        self._all_items = tuple(all_items)
        self._requests = tuple(sorted(bundle_items))

    def __repr__(self):
        return "Bundle(" + ", ".join(str(x.uid) for x in self.requests) + ")"

    def __len__(self):
        return len(self.requests)

    def __eq__(self, other):
        return self._requests == other._requests

    def __hash__(self):
        return hash(self.requests)

    def __conform__(self, protocol):
        if protocol is sqlite3.PrepareProtocol:
            return ";".join(str(x) for x in self.requests)

    def __iter__(self):
        raise NotImplementedError(
            "Bundles have two representations of their elements: requests and vertices. "
            "Iterate over those directly."
        )

    @property
    def all_items(self):
        return self._all_items

    @property
    def requests(self):
        return self._requests

    @cached_property
    def bitstring(self) -> tuple[int, ...]:
        return tuple(1 if x in self.requests else 0 for x in self.all_items)

    @cached_property
    def vertices(self):
        return tuple(request.uid for request in self.requests)

    @classmethod
    def from_binary(cls, request_pool, bitstring) -> "Bundle":
        bundle_items = tuple(
            request_pool[i] for i in range(len(bitstring)) if bitstring[i] == 1
        )
        return cls(request_pool, bundle_items)

    @classmethod
    def random(cls, request_pool: Sequence[Request]):
        """
        Returns a random subset of requests. The empty bundle is not included in the set of possible bundles.
        https://stackoverflow.com/a/3947856/15467861
        """
        num_subsets = 2 ** len(request_pool)
        rand = random.randint(1, num_subsets - 1)  # exclude empty set
        rand = bin(rand)[2:].zfill(len(request_pool))
        requests = [request_pool[i] for i, x in enumerate(rand) if x == "1"]
        return cls(request_pool, requests)

    @staticmethod  # TODO shouldn't this be class-level attributes instead of staticmethods?
    def _global_features_names():
        return [
            "cardinality",
            "spatial_centroid_x",
            "spatial_centroid_y",
            "GH_density",
            "tour_sum_travel_duration",
            "mean_vertex_center_angles",
            "std_vertex_center_angles",
            "min_x_coord",
            "max_x_coord",
            "mean_x_coord",
            "min_y_coord",
            "max_y_coord",
            "mean_y_coord",
            "min_time_window_center",
            "max_time_window_center",
            "mean_time_window_center",
            # ChatGPT suggested
            "bounding_box_area",
            "bounding_box_aspect_ratio",
            "convex_hull_area",
            "convex_hull_perimeter",
            "min_pairwise_duration",
            "max_pairwise_duration",
            "sum_pairwise_duration",
            "sum_of_squared_pairwise_duration",
            "mean_pairwise_duration",
            "std_pairwise_duration",
            "min_pairwise_time_window_difference",
            "max_pairwise_time_window_difference",
            "sum_pairwise_time_window_difference",
            "sum_of_squared_pairwise_time_window_difference",
            "mean_pairwise_time_window_difference",
            "std_pairwise_time_window_difference",
            "min_centroid_distance",
            "max_centroid_distance",
            "sum_centroid_distance",
            "sum_of_squared_centroid_distance",
            "mean_centroid_distance",
            "std_centroid_distance",
            "min_nearest_neighbor_duration",
            "max_nearest_neighbor_duration",
            "sum_nearest_neighbor_duration",
            "sum_of_squared_nearest_neighbor_duration",
            "mean_nearest_neighbor_duration",
            "std_nearest_neighbor_duration",
            "std_nearest_neighbor_angle",
            "nearest_neighbor_duration_ratio",
            "density",
            "convex_hull_solidity",
        ]

    @staticmethod  # TODO shouldn't this be class-level attributes instead of staticmethods?
    def _carrier_specific_features_names():
        return [
            "num_requests_from_carrier",
            "num_requests_not_from_carrier",
            "fraction_requests_from_carrier",
            "fraction_requests_not_from_carrier",
            "min_duration_to_carrier_depot",
            "max_duration_to_carrier_depot",
            "sum_duration_to_carrier_depot",
            "sum_of_squared_duration_to_carrier_depot",
            "mean_duration_to_carrier_depot",
            "std_duration_to_carrier_depot",
            "tour_with_depot_sum_travel_duration",
            "mean_vertex_carrier_depot_angle",
            "std_vertex_carrier_depot_angle",
        ]

    def global_features(self):
        """return a dictionary of all global features of the bundle. Global features are those that are not specific
        to a single carrier, e.g. cardinality, density, the sum of pairwise durations."""
        global_features = {
            feature: getattr(self, feature) for feature in self._global_features_names()
        }
        return global_features

    def carrier_specific_features(self, carrier_id: int):
        """return a dictionary of all carrier-specific features of the bundle. Carrier-specific features are those that
        are specific to a single carrier."""
        carrier_specific_features = {
            feature: getattr(self, feature)(carrier_id)
            for feature in self._carrier_specific_features_names()
        }
        return carrier_specific_features

    @cached_property
    def spatial_centroid(self):
        """tuple of x and y coordinates of the bundle's centroid.
        Note: spatial only, does not consider temporal dimension"""
        x = [request.x for request in self.requests]
        y = [request.y for request in self.requests]
        return sum(x) / len(x), sum(y) / len(y)

    @cached_property
    def spatial_centroid_x(self):
        return self.spatial_centroid[0]

    @cached_property
    def spatial_centroid_y(self):
        return self.spatial_centroid[1]

    @cached_property
    def centroid_euclidean_distances(self) -> dict[Request, float]:
        """
        Note: spatial only, does not consider temporal dimension

        :return: dictionary of Euclidean distances to the bundle's centroid with vertex IDs as keys
        """
        centroid_euclidean_distances = dict()
        for request in self.requests:
            distance = ut.euclidean_distance(
                request.x, request.y, *self.spatial_centroid
            )
            centroid_euclidean_distances[request] = distance
        return centroid_euclidean_distances

    def spatial_medoid(self, instance: CAHDInstance):
        """
        Note: spatial only, does not consider temporal dimension

        :param instance:
        :return the medoid request
        """
        min_dur = dt.timedelta.max
        medoid = None

        for request_0 in self.requests:
            duration = dt.timedelta(0)
            for request_1 in self.requests:
                if request_0 == request_1:
                    continue
                duration += instance.travel_duration([request_0], [request_1])
            if duration < min_dur:
                min_dur = duration
                medoid = request_0
        return medoid

    @cached_property
    def GH_radius(self):
        """
        average Euclidean distance of all points in the bundle to the bundle's centroid

        Note: spatial only, does not consider temporal dimension
        The radius of a bundle is defined as the average distance of all points in the bundle to the bundle's centroid
        :return:
        """
        distances = list(self.centroid_euclidean_distances.values())
        GH_radius = sum(distances) / len(distances)
        return GH_radius

    @cached_property
    def GH_density(self):
        """
        average request-to-centroid pendulum distance divided by the maximum request-to-centroid pendulum distance

        """
        distances = list(self.centroid_euclidean_distances.values())
        if (max_density := max(distances)) == 0:
            GH_density = 0
        else:
            GH_density = mean(distances) / max_density
        return GH_density

    def GH_separation(self, other):
        """
        Note: spatial only, does not consider temporal dimension
        """
        other: Bundle
        centroid_dist = ut.euclidean_distance(
            *self.spatial_centroid, *other.spatial_centroid
        )
        max_radius = max(self.GH_radius, other.GH_radius)
        if max_radius > 0:
            return centroid_dist / max_radius
        else:
            return centroid_dist

    def GH_isolation(self, others):
        """
        Note: spatial only, does not consider temporal dimension
        isolation is defined as the minimum of all its separation values to others
        :param others:
        :return:
        """
        others: Sequence[Bundle]
        isolation = float("inf")
        for other in others:
            separation = self.GH_separation(other)
            if separation < isolation:
                isolation = separation
        return isolation

    def tour(self, instance, routing_solver) -> Optional[Tour]:
        """
        Construct a tour for the bundle. The tour is constructed by inserting the requests in the bundle into a tour
        in the order of their earliest disclosure time. The tour construction is performed by the
        VRPTWMinTravelDurationInsertion heuristic.

        :param instance:
        :param routing_solver:
        :return:
        """
        raise NotImplementedError(
            "After the RoutingSolver change, this is not functional anymore"
        )
        # treat the first-to-open vertex as the depot
        depot_request = min(self.requests, key=lambda x: x.disclosure_time)
        depot = Depot(
            -1,
            depot_request.uid,
            depot_request.x,
            depot_request.y,
            depot_request.tw_open,
            depot_request.tw_close,
        )
        return tour

    def tour_with_depot(
        self, instance: CAHDInstance, depot: Depot, routing_solver
    ) -> Optional[Tour]:
        raise NotImplementedError(
            "After the RoutingSolver change, this is not functional anymore"
        )
        return tour

    def _travel_duration_matrix(self, instance) -> np.ndarray:
        """pairwise travel duration matrix for all vertices in the bundle
        :param instance:
        """
        return instance._travel_duration_matrix[np.ix_(self.vertices, self.vertices)]

    def _time_window_distance_matrix(self, instance) -> np.ndarray:
        return instance._time_window_distance_matrix[
            np.ix_(self.vertices, self.vertices)
        ]

    def tour_sum_travel_duration(self, instance: CAHDInstance, routing_solver):
        """total travel duration of the bundle's tour in seconds
        :param routing_solver:
        :param instance:
        """
        if isinstance(x := self.tour(instance, routing_solver), Tour):
            return x.sum_travel_duration.total_seconds()
        else:
            # return nan value that is also writable to sqlite3
            return float("nan")

    @cached_property
    def min_time_window_center(self) -> float:
        """earliest time window center of all bundle members. returned as timedelta since datetime.datetime.min"""
        min_tw_center = min(
            TimeWindow(request.tw_open, request.tw_close).center
            for request in self.requests
        )
        return (min_tw_center - dt.datetime.min).total_seconds()

    @cached_property
    def max_time_window_center(self) -> float:
        """latest time window center of all bundle members. returned as timedelta since datetime.datetime.min"""
        max_tw_center = max(
            TimeWindow(request.tw_open, request.tw_close).center
            for request in self.requests
        )
        return (max_tw_center - dt.datetime.min).total_seconds()

    @cached_property
    def mean_time_window_center(self) -> float:
        """mean of the time window centers of all bundle members. returned as timedelta since datetime.datetime.min"""
        time_window_centers = [
            TimeWindow(request.tw_open, request.tw_close).center
            for request in self.requests
        ]
        time_window_center_timedelta = [
            x - EXECUTION_START_TIME for x in time_window_centers
        ]
        mean_timedelta = sum(time_window_center_timedelta, dt.timedelta(0)) / len(
            time_window_centers
        )
        mean_tw_center = EXECUTION_START_TIME + mean_timedelta
        return (mean_tw_center - dt.datetime.min).total_seconds()

    @cached_property
    def min_pairwise_time_window_difference(self):
        """minimum of all absolute pairwise time window differences between bundle members. The difference between two time
        windows is defined as the absolute difference of their centers.
        Example: tw0 = [8:00, 10:00], tw1 = [9:00, 11:00] => difference = 1 hour"""
        if self.cardinality <= 1:
            return 0
        if not np.any(self._time_window_distance_matrix > dt.timedelta(0)):
            return 0
        return (
            self._time_window_distance_matrix[
                self._time_window_distance_matrix > dt.timedelta(0)
            ]
            .min()
            .total_seconds()
        )

    @cached_property
    def max_pairwise_time_window_difference(self):
        """maximum of all pairwise time window differences between bundle members. The difference between two time
        windows is defined as the absolute difference of their centers.
        Example: tw0 = [8:00, 10:00], tw1 = [9:00, 11:00] => difference = 1 hour"""
        if self.cardinality <= 1:
            return 0
        return self._time_window_distance_matrix.max().total_seconds()

    @cached_property
    def sum_pairwise_time_window_difference(self):
        """sum of all ABSOLUTE pairwise time window differences between bundle members. The difference between two time
        windows is defined as the absolute difference of their centers.
        Example: tw0 = [8:00, 10:00], tw1 = [9:00, 11:00] => difference = 1 hour"""
        if self.cardinality <= 1:
            return 0
        # divide by 2 because the matrix is symmetric and the diagonal is zero
        return np.abs(self._time_window_distance_matrix).sum().total_seconds() / 2

    @cached_property
    def sum_of_squared_pairwise_time_window_difference(self):
        """sum of all squared pairwise time window differences between bundle members. The difference between two time
        windows is defined as the absolute difference of their centers.
        Example: tw0 = [8:00, 10:00], tw1 = [9:00, 11:00] => difference = 1 hour"""
        if self.cardinality <= 1:
            return 0
        matrix = np.vectorize(lambda x: x.total_seconds())(
            self._time_window_distance_matrix
        )
        # divide by 2 because the matrix is symmetric and the diagonal is zero
        return np.square(matrix).sum() / 2

    @cached_property
    def mean_pairwise_time_window_difference(self):
        """mean of all pairwise time window differences between bundle members. The difference between two time
        windows is defined as the absolute difference of their centers.
        Example: tw0 = [8:00, 10:00], tw1 = [9:00, 11:00] => difference = 1 hour"""
        if self.cardinality <= 1:
            return 0
        return np.mean(np.abs(self._time_window_distance_matrix)).total_seconds()

    @cached_property
    def std_pairwise_time_window_difference(self):
        """std of all pairwise time window differences between bundle members. The difference between two time
        windows is defined as the absolute difference of their centers.
        Example: tw0 = [8:00, 10:00], tw1 = [9:00, 11:00] => difference = 1 hour"""
        if self.cardinality <= 1:
            return 0
        mean_tw_diff = self.mean_pairwise_time_window_difference
        matrix = np.vectorize(lambda x: x.total_seconds())(
            self._time_window_distance_matrix
        )
        return np.sqrt(np.sum(np.square(matrix - mean_tw_diff)) / self.cardinality)

    def spatio_temporal_graph_cohesion(
        self, spatial_weight: float = 0.5, temporal_weight: float = 0.5
    ):
        """
        graph-based cohesion, i.e. sum of pairwise distance/dissimilarity of bundle members.
        :return:
        """
        assert spatial_weight + temporal_weight == 1.0

        cohesion = 0
        norm_tw_dist_denom = (
            self.instance.max_abs_time_window_distance
            - self.instance.min_abs_time_window_distance
        )
        norm_travel_dur_denom = (
            self.instance.max_travel_duration - self.instance.min_travel_duration
        )
        for i, request_0 in enumerate(self.requests[:-1]):
            for request_1 in self.requests[
                i + 1 :
            ]:  # TODO is it really sufficient to start from i+1? Or better do all?
                if norm_tw_dist_denom == dt.timedelta(0):
                    norm_tw_dist = 0
                else:
                    tw_dist = abs(
                        TimeWindow(request_0.tw_open, request_0.tw_close).center
                        - TimeWindow(request_1.tw_open, request_1.tw_close).center
                    )
                    norm_tw_dist = (
                        tw_dist - self.instance.min_abs_time_window_distance
                    ) / norm_tw_dist_denom

                travel_dur = self.instance.travel_duration([request_0], [request_1])
                norm_travel_dur = (
                    travel_dur - self.instance.min_travel_duration
                ) / norm_travel_dur_denom

                cohesion += (
                    spatial_weight * norm_travel_dur + temporal_weight * norm_tw_dist
                )
        return cohesion

    @cached_property
    def LS_spatio_temporal_cohesion(self) -> float:
        """
        Los, Schulte et al. (2020) define a bundle's quality by the sum of pairwise spatio-temporal dissimilarities.
        These dissimilarities are given by a weighted sum of (1) travel duration between the requests and (2) the
        minimal waiting time (due to time window restrictions) at one of the locations if a vehicle serves both
        locations immediately after each other.

        :return:
        """
        if len(self) <= 1:
            return 0
        else:
            pairwise_dissimilarity = []
            for i, v0 in enumerate(self.vertices[:-1]):
                for v1 in self.vertices[i + 1 :]:
                    dissimilarity = self.instance.LS_vertex_dissimilarity_matrix[v0][v1]
                    pairwise_dissimilarity.append(dissimilarity)
            return sum(pairwise_dissimilarity) / len(pairwise_dissimilarity)

    def spatial_depot_duration(self, depot: Depot):
        """sum of travel durations from a request to a given carrier depot and back"""
        dist = dt.timedelta(0)
        for request in self.requests:
            dist += self.instance.travel_duration([request], [depot])
            dist += self.instance.travel_duration([depot], [request])
        return dist

    def travel_durations_to_vertex(self, vertex: int):
        """return the average of the two durations [request, vertex] and [vertex, request] for all requests. In other
        words, it returns a list of the 'average' pendulum durations for all requests in the bundle."""
        travel_durations = []
        for request in self.requests:
            travel_durations.append(
                self.instance.travel_duration([request, vertex], [vertex, request]) / 2
            )
        return travel_durations

    @cached_property
    def avg_tw_group_spatial_graph_cohesion(self):
        """
        Note: spatial only, does not consider temporal dimension
        Requests in the bundle are grouped by their time window. The spatial cohesion of all groups is computed and
        the average of the cohesion values is returned
        :return:
        """
        tw_groups = {
            TimeWindow(request.tw_open, request.tw_close) for request in self.requests
        }
        tw_groups = {
            k: [r for r in self.requests if TimeWindow(r.tw_open, r.tw_close) == k]
            for k in tw_groups
        }
        tw_groups_cohesion = dict()
        for tw, request_group in tw_groups.items():
            cohesion = 0
            for r0 in request_group:
                for r1 in request_group:
                    cohesion += self.instance.travel_duration(
                        [r0], [r1]
                    ).total_seconds()
            tw_groups_cohesion[tw] = cohesion
        return mean(tw_groups_cohesion.values())

    @cached_property
    def vertex_center_angles(self) -> Sequence[float]:
        """the angles between the request locations and the city center"""
        assert "vienna" in self.instance.meta["t"]
        vienna_center = (48.210033, 16.363449)
        angles = []
        for request in self.requests:
            a = compute_angle_degrees(vienna_center, (request.x, request.y))
            angles.append(a)
        return angles

    @cached_property
    def mean_vertex_center_angles(self) -> float:
        return mean(self.vertex_center_angles)

    @cached_property
    def std_vertex_center_angles(self) -> float:
        if self.cardinality <= 1:
            return 0
        return stdev(self.vertex_center_angles)

    @cached_property
    def min_x_coord(self):
        return min(request.x for request in self.requests)

    @cached_property
    def max_x_coord(self):
        return max(request.x for request in self.requests)

    @cached_property
    def mean_x_coord(self):
        return mean(request.x for request in self.requests)

    @cached_property
    def min_y_coord(self):
        return min(request.y for request in self.requests)

    @cached_property
    def max_y_coord(self):
        return max(request.y for request in self.requests)

    @cached_property
    def mean_y_coord(self):
        return mean(request.y for request in self.requests)

    # --------------------- AI feature suggestions ---------------------
    @cached_property
    def cardinality(self):
        """The number of points in the set."""
        return len(self.requests)

    # @cached_property
    # def centroid(self):
    #     """The center of mass or average position of all the points."""
    #     pass

    @cached_property
    def bounding_box_area(self):
        """The area of the smallest rectangle that can contain all the points."""
        width = self.max_x_coord - self.min_x_coord
        height = self.max_y_coord - self.min_y_coord
        return width * height

    @cached_property
    def bounding_box_aspect_ratio(self):
        """The ratio of the longer side to the shorter side of the bounding box."""
        if self.cardinality <= 1:
            return 0
        width = self.max_x_coord - self.min_x_coord
        height = self.max_y_coord - self.min_y_coord
        return max(width, height) / min(width, height)

    # NOTE: this would be a metric feature with only 0 or 90 values. Not good.
    # @cached_property
    # def orientation_of_the_bounding_box(self):
    #     """The angle between the longer side of the bounding box and a reference axis."""
    #     width = self.max_x_coord - self.min_x_coord
    #     height = self.max_y_coord - self.min_y_coord
    #     if width > height:
    #         return 0
    #     else:
    #         return 90

    @cached_property
    def _convex_hull(self):
        """The smallest convex polygon that encloses all the points."""
        return ConvexHull([(request.x, request.y) for request in self.requests])

    @cached_property
    def convex_hull_area(self):
        """The area of the smallest convex polygon that encloses all the points."""
        if self.cardinality <= 2:
            return 0
        return self._convex_hull.volume

    @cached_property
    def convex_hull_perimeter(self):
        """The perimeter of the smallest convex polygon that encloses all the points."""
        if self.cardinality <= 2:
            return 0  # alternatively, could also return 2*(distance between the two points)?!
        return self._convex_hull.area

    @cached_property
    def min_pairwise_duration(self):
        """The minimum duration between any pair of points in the set."""
        if self.cardinality <= 1:
            return 0
        return (
            self._travel_duration_matrix[self._travel_duration_matrix > dt.timedelta(0)]
            .min()
            .total_seconds()
        )

    @cached_property
    def max_pairwise_duration(self):
        """The maximum duration between any two points in the set."""
        return self._travel_duration_matrix.max().total_seconds()

    @cached_property
    def sum_pairwise_duration(self):
        """The sum of the durations between all pairs of points in the set."""
        return self._travel_duration_matrix.sum().total_seconds()

    @cached_property
    def sum_of_squared_pairwise_duration(self):
        """The sum of the squared durations between all pairs of points in the set."""
        matrix = np.vectorize(lambda x: x.total_seconds())(self._travel_duration_matrix)
        return (matrix**2).sum()

    @cached_property
    def mean_pairwise_duration(self):
        """The average duration between all pairs of points in the set."""
        return self._travel_duration_matrix.mean().total_seconds()

    @cached_property
    def std_pairwise_duration(self):
        """A measure of the spread of durations between points in the set."""
        return np.vectorize(lambda x: x.total_seconds())(
            self._travel_duration_matrix
        ).std()

    # NOTE: too complex to compute
    # @cached_property
    # def area_minimum_enclosing_circle(self):
    #     """The area of the smallest circle that encloses all the points."""
    #     pass

    # @cached_property
    # def perimeter_of_the_smallest_enclosing_circle(self):
    #     """The perimeter of the smallest circle that encloses all the points."""
    #     pass
    @cached_property
    def min_centroid_distance(self):
        """The minimum of the Euclidean distances of each point to the centroid."""
        if self.cardinality <= 1:
            return 0
        distances = list(self.centroid_euclidean_distances.values())
        return min(distances)

    @cached_property
    def max_centroid_distance(self):
        """The maximum of the Euclidean distances of each point to the centroid."""
        if self.cardinality <= 1:
            return 0
        distances = list(self.centroid_euclidean_distances.values())
        return max(distances)

    @cached_property
    def sum_centroid_distance(self):
        """The sum of the Euclidean distances of each point to the centroid."""
        if self.cardinality <= 1:
            return 0
        distances = list(self.centroid_euclidean_distances.values())
        return sum(distances)

    @cached_property
    def sum_of_squared_centroid_distance(self):
        """The sum of the squared Euclidean distances of each point to the centroid."""
        if self.cardinality <= 1:
            return 0
        distances = list(self.centroid_euclidean_distances.values())
        return sum([d**2 for d in distances])

    @cached_property
    def mean_centroid_distance(self):
        """The average distance of each point to the centroid. Same as the 'radius' feature suggested by Gansterer &
        Hartl (2018?)"""
        distances = list(self.centroid_euclidean_distances.values())
        return mean(distances)

    @cached_property
    def std_centroid_distance(self):
        """A measure of the spread of distances between points and the centroid."""
        if self.cardinality <= 1:
            return 0
        centroid_distances = list(self.centroid_euclidean_distances.values())
        return stdev(centroid_distances)

    @cached_property
    def min_nearest_neighbor_duration(self):
        """The minimum distance to the nearest neighbor for each point."""
        if self.cardinality <= 1:
            return 0
        nearest_neighbor_durations = self._nearest_neighbor_durations
        return nearest_neighbor_durations.min().total_seconds()

    @cached_property
    def _nearest_neighbor_durations(self):
        nearest_neighbors = np.argsort(self._travel_duration_matrix, axis=1)[:, 1]
        nearest_neighbor_durations = self._travel_duration_matrix[
            np.arange(len(nearest_neighbors)), nearest_neighbors
        ]
        return nearest_neighbor_durations

    @cached_property
    def max_nearest_neighbor_duration(self):
        """The maximum distance to the nearest neighbor for each point."""
        if self.cardinality <= 1:
            return 0
        return self._nearest_neighbor_durations.max().total_seconds()

    @cached_property
    def sum_nearest_neighbor_duration(self):
        """The sum of the distances to the nearest neighbor for each point."""
        if self.cardinality <= 1:
            return 0
        return self._nearest_neighbor_durations.sum().total_seconds()

    @cached_property
    def sum_of_squared_nearest_neighbor_duration(self):
        """The sum of the squared distances to the nearest neighbor for each point."""
        if self.cardinality <= 1:
            return 0
        nearest_neighbor_durations = np.vectorize(lambda x: x.total_seconds())(
            self._nearest_neighbor_durations
        )
        return (nearest_neighbor_durations**2).sum()

    @cached_property
    def mean_nearest_neighbor_duration(self):
        """The average distance to the nearest neighbor for each point."""
        if self.cardinality <= 1:
            return 0
        return np.mean(self._nearest_neighbor_durations).total_seconds()

    @cached_property
    def std_nearest_neighbor_duration(self):
        """A measure of the spread of distances between points and their nearest neighbors."""
        if self.cardinality <= 1:
            return 0
        nearest_neighbor_durations = np.vectorize(lambda x: x.total_seconds())(
            self._nearest_neighbor_durations
        )
        return nearest_neighbor_durations.std()

    @cached_property
    def nearest_neighbor_duration_ratio(self):
        """A measure of the degree to which the points form clusters or groups: ratio between the average nearest
        neighbor travel duration and the expected average nearest neighbor travel duration in a random distribution
        with the same number of points. A ratio greater than 1 suggests clustering"""
        inst_nearest_neighbors_durations = self.instance.nearest_neighbors_durations
        return (
            self.mean_nearest_neighbor_duration
            / inst_nearest_neighbors_durations.mean().total_seconds()
        )

    @cached_property
    def density(self):
        """The number of points divided by the area of the bounding box."""
        if self.cardinality <= 1:
            return 1  # NOTE: simplification, for a single point, density really loses its meaning
        return self.cardinality / self.bounding_box_area

    # NOTE: not a very useful feature
    # @cached_property
    # def convexity(self):
    #     """A measure of how concave or convex the set of points is.
    #     Calculate the area of the convex hull of the points using the scipy.spatial module.
    #     Calculate the total area enclosed by the points by computing the area of the bounding box around the points.
    #     Compute the convexity as the ratio of the convex hull area to the total area enclosed by the points.
    #     """
    #     pass

    # NOTE: too complex to compute
    # def eccentricity(self):
    #     """A measure of how elongated the set of points is."""
    #     pass

    @cached_property
    def convex_hull_solidity(self):
        """A measure of how convex the set of points is. Ratio of the area of the convex hull to the area of the
        bounding box."""
        if self.cardinality <= 1:
            return 1
        return self.convex_hull_area / self.bounding_box_area

    @cached_property
    def std_nearest_neighbor_angle(self):
        """A measure of the spread of the angles between the nearest neighbor vectors."""
        if self.cardinality <= 1:
            return 0
        nearest_neighbor_angles = []
        for i, v in enumerate(self.vertices):
            nearest_neighbor_vertex_index = np.argsort(self._travel_duration_matrix[i])[
                1
            ]  # skip 0 duration to v itself
            nearest_neighbor_vertex = self.vertices[nearest_neighbor_vertex_index]
            nearest_neighbor_angle = compute_angle_degrees(
                self.instance.coords(v), self.instance.coords(nearest_neighbor_vertex)
            )
            nearest_neighbor_angles.append(nearest_neighbor_angle)
        return stdev(nearest_neighbor_angles)

    """
    Additional global feautres to consider:
    Nearest neighbor angle variability: A measure of how variable the angles between the nearest neighbor vectors are.
    Nearest neighbor orientation: The average orientation of the nearest neighbor vectors.
    Convex hull vertex count: The number of vertices in the convex hull polygon.  (probably best to take the relative)
    
    """

    def num_requests_from_carrier(self, carrier_id: int) -> int:
        """Returns the number of requests from the bundle that belonged to carrier with id carrier_id originally, i.e.
        before request selection."""
        num_requests_from_carrier = 0
        for request in self.requests:
            if request.initial_carrier_assignment == carrier_id:
                num_requests_from_carrier += 1
        return num_requests_from_carrier

    def num_requests_not_from_carrier(self, carrier_id: int) -> int:
        return self.cardinality - self.num_requests_from_carrier(carrier_id)

    def fraction_requests_from_carrier(self, carrier_id: int) -> float:
        return self.num_requests_from_carrier(carrier_id) / self.cardinality

    def fraction_requests_not_from_carrier(self, carrier_id: int) -> float:
        return self.num_requests_not_from_carrier(carrier_id) / self.cardinality

    def min_duration_to_carrier_depot(self, carrier_id: int) -> float:
        """Returns the minimum travel duration from any request in the bundle to the carrier's depot."""
        return min(self.travel_durations_to_vertex(carrier_id)).total_seconds()

    def max_duration_to_carrier_depot(self, carrier_id: int) -> float:
        """Returns the maximum travel duration from any request in the bundle to the carrier's depot."""
        return max(self.travel_durations_to_vertex(carrier_id)).total_seconds()

    def sum_duration_to_carrier_depot(self, carrier_id: int) -> float:
        """Returns the sum of the travel durations from any request in the bundle to the carrier's depot."""
        duration_to_carrier_depot = self.travel_durations_to_vertex(carrier_id)
        return sum([d.total_seconds() for d in duration_to_carrier_depot])

    def sum_of_squared_duration_to_carrier_depot(self, carrier_id: int) -> float:
        """Returns the sum of squared errors of the travel duration from any request in the bundle to the carrier's
        depot."""
        duration_to_carrier_depot = self.travel_durations_to_vertex(carrier_id)
        return sum(np.square([d.total_seconds() for d in duration_to_carrier_depot]))

    def mean_duration_to_carrier_depot(self, carrier_id: int) -> float:
        """Returns the mean travel duration from any request in the bundle to the carrier's depot."""
        duration_to_carrier_depot = self.travel_durations_to_vertex(carrier_id)
        return mean([d.total_seconds() for d in duration_to_carrier_depot])

    def std_duration_to_carrier_depot(self, carrier_id: int) -> float:
        """Returns the standard deviation of the travel duration from any request in the bundle to the carrier's
        depot."""
        if self.cardinality <= 1:
            return 0
        duration_to_carrier_depot = self.travel_durations_to_vertex(carrier_id)
        return stdev([d.total_seconds() for d in duration_to_carrier_depot])

    def tour_with_depot_sum_travel_duration(
        self, instance: CAHDInstance, carrier_id: int, routing_solver
    ):
        """Returns the sum of the travel durations from the carrier's depot to all requests in the bundle and back to
        the depot.
        :param routing_solver:
        :param instance:"""
        if isinstance(
            t := self.tour_with_depot(
                instance, instance.depots[carrier_id], routing_solver
            ),
            Tour,
        ):
            return t.sum_travel_duration.total_seconds()
        else:
            return float("nan")

    def vertex_carrier_depot_angles(
        self, instance: CAHDInstance, carrier_id: int
    ) -> list[float]:
        """Returns the angles between the carrier's depot and all requests in the bundle.
        :param instance:
        """
        angles = []
        depot = instance.depots[carrier_id]
        for request in self.requests:
            angle = compute_angle_degrees((depot.x, depot.y), (request.x, request.y))
            angles.append(angle)
        return angles

    def min_vertex_carrier_depot_angle(
        self, instance: CAHDInstance, carrier_id: int
    ) -> float:
        """Returns the minimum angle between the carrier's depot and all requests in the bundle.
        :param instance:
        """
        return min(self.vertex_carrier_depot_angles(instance, carrier_id))

    def max_vertex_carrier_depot_angle(self, instance, carrier_id: int) -> float:
        """Returns the maximum angle between the carrier's depot and all requests in the bundle.
        :param instance:
        """
        return max(self.vertex_carrier_depot_angles(instance, carrier_id))

    def mean_vertex_carrier_depot_angle(self, instance, carrier_id: int) -> float:
        """Returns the mean angle between the carrier's depot and all requests in the bundle.
        :param instance:
        """
        return mean(self.vertex_carrier_depot_angles(instance, carrier_id))

    def std_vertex_carrier_depot_angle(self, instance, carrier_id: int) -> float:
        """Returns the standard deviation of the angles between the carrier's depot and all requests in the bundle.
        :param instance:
        """
        if self.cardinality <= 1:
            return 0
        return stdev(self.vertex_carrier_depot_angles(instance, carrier_id))

    def ruether_rieck(self, carrier_id: int) -> float:
        """
        Returns the Ruether & Rieck (2021) score for the bundle and carrier combination. The score is defined as:
        ((sum of revenues) * (cardinality) * (num vehicles required to serve bundle) * (density)) /
        ((duration of routes required to serve bundle) * (duration to carrier depot))

        All parts are weighted using custom weights:
        - sum of revenues: 0.15
        - cardinality: 1.8
        - num vehicles required to serve bundle: 0.2
        - density: 0.0
        - duration of routes required to serve bundle: 0.0
        - duration to carrier depot: 1.5
        """
        raise NotImplementedError(
            "After the RoutingSolver change, this is not functional anymore"
        )

        def required_tours() -> list[Tour]:
            """
            Returns the tours required to serve all requests in the bundle. The tours are constructed using the
            VRPTWMinTravelDurationInsertion heuristic.
            """
            routing_solver = None
            tours = []
            for request in sorted(self.requests, key=lambda x: x.disclosure_time):
                best_delta = float("inf")
                best_pos = None
                best_tour: Tour = None

                # test the existing tours
                for tour in tours:
                    delta, pos = routing_solver.best_insertion_for_request_in_tour(
                        self.instance, tour, request, check_feasibility=True
                    )
                    if delta < best_delta:
                        best_delta = delta
                        best_pos = pos
                        best_tour = tour

                # test a new tour
                depot = self.instance.depots[carrier_id]
                tmp_tour = Tour("tmp", depot)
                delta, pos = routing_solver.best_insertion_for_request_in_tour(
                    self.instance, tmp_tour, request, check_feasibility=True
                )
                # NOTE: delta = delta * 0.75  # to facilitate opening new tours,
                #  only .75 of the pendulum's delta is accounted for
                if delta < best_delta:
                    best_delta = delta
                    best_tour = tmp_tour
                    best_pos = pos

                # insert the request into the best tour
                best_tour.insert_and_update(self.instance, [best_pos], [request])
                if best_tour not in tours:
                    tours.append(best_tour)

            return tours

        alpha = 0.15
        beta = 1.8
        gamma = 0.2
        mu = 0.0
        v = 0.0
        tau = 1.5

        sum_of_revenues = sum(r.revenue for r in self.requests)
        cardinality = self.cardinality
        tours = required_tours()
        num_vehicles_required = len(tours)
        density = self.density

        duration_of_tours = sum(t.sum_travel_duration.total_seconds() for t in tours)
        duration_to_carrier_depot = self.sum_duration_to_carrier_depot(carrier_id)

        numerator = (
            sum_of_revenues**alpha
            * cardinality**beta
            * num_vehicles_required**gamma
            * density**mu
        )
        denominator = duration_of_tours**v * duration_to_carrier_depot**tau
        result = numerator / denominator
        return result

    def get_feature(self, feature: str):
        """
        take a string and return the corresponding feature value.
        feature argument should be of the form 'global_feature_x' or 'carrier_specific_feature_x' where x is the name
        of the feature.
        """
        if feature.startswith("global_feature_"):
            return self.__getattribute__(feature.removeprefix("global_feature_"))
        elif feature.startswith("carrier_specific_feature_"):
            raw_feature, carrier_id = feature.removeprefix(
                f"carrier_specific_feature_"
            ).rsplit("_", 1)
            return self.__getattribute__(raw_feature)(int(carrier_id))
        else:
            raise ValueError(
                f'Invalid feature name: {feature}. (Remember to use the correct prefix "global_feature_" '
                f'or "carrier_specific_feature_".)'
            )

    def as_record(self, instance):
        """
        returns a dict of all features and the bundle's str representation

        :param instance:
        :return:
        """
        record = dict()
        record.update(instance.meta)
        record["bundle"] = ";".join(str(x) for x in self.requests)
        # global features
        global_bundle_features_dict = self.global_features()
        record.update(
            {f"global_feature_{k}": v for k, v in global_bundle_features_dict.items()}
        )
        # carrier specific features
        for carrier_id in range(instance.num_carriers):
            carrier_specific_bundle_features_dict = self.carrier_specific_features(
                carrier_id
            )
            record.update(
                {
                    f"carrier_specific_feature_{k}_{carrier_id}": v
                    for k, v in carrier_specific_bundle_features_dict.items()
                }
            )
        return record

    def to_img(self, instance, res: 128):
        """
        returns a 2d array of shape (res, res) of the bundle. O entries are empty, 1 entries means there is a
        request in that location
        :param instance:
        :param res:
        :return:
        """
        img = np.zeros((res, res))
        for request in self.requests:
            x, y = request.x, request.y
            x = int(
                (x - instance.min_x_coord)
                / (instance.max_x_coord - instance.min_x_coord)
                * res
            )
            y = int(
                (y - instance.min_y_coord)
                / (instance.max_y_coord - instance.min_y_coord)
                * res
            )
            img[x, y] = 1
        return img

    def plot(self, ax: plt.Axes = None):
        """
        Plots the bundle on a given matplotlib axis. If no axis is given, a new one is created.
        :param ax: matplotlib axis
        :return:
        """
        if ax is None:
            fig, ax = plt.subplots()
        for request in self.requests:
            ax.scatter(request.x, request.y)
        return ax


def random_bundle(n) -> tuple[int]:
    """
    generate a random bundle from n items
    :param n: number of items
    :return: binary vector of size n
    """
    from utility_module.random import rng

    return tuple(rng.choice([0, 1], size=n))


def random_bundle_max_k(n, k) -> tuple[int]:
    """
    generate a random bundle from n items that covers at most k items
    :param n: number of items
    :param k: maximum number of items to cover
    :return: binary vector of size n
    """
    from utility_module.random import rng

    bundle = [0] * n
    k_prob = np.array([math.comb(n, i) / 2**n for i in range(k)])
    k_prob = k_prob / k_prob.sum()
    selected_k = rng.choice(range(1, k + 1), p=k_prob)
    selected_indices = rng.choice(range(n), size=selected_k, replace=False)
    for i in selected_indices:
        bundle[i] = 1
    return tuple(bundle)
