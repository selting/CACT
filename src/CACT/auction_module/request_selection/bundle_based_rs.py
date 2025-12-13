import datetime as dt
import itertools
from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Sequence

import numpy as np

import core_module.carrier
from auction_module.bundle_generation.bundle_based.bundle import Bundle
from auction_module.request_selection.request_selection import RequestSelectionStrategy
from core_module import instance as it, solution as slt


class RequestSelectionStrategyBundle(RequestSelectionStrategy, ABC):
    """
    Select (for each carrier) a set of requests based on their *combined* evaluation of a given measure (e.g. similarity
    of set members). This idea of bundled evaluation is also based on Gansterer & Hartl (2016)
    """
    def preprocessing(self, instance: it.CAHDInstance, solution: slt.CAHDSolution):
        pass

    @abstractmethod
    def evaluate_bundle(self, instance: it.CAHDInstance, carrier: core_module.carrier.Carrier, bundle: Bundle):
        # TODO It could literally be a bundle_valuation strategy that is executed here. Not a partition_valuation though
        pass

    def valuations_post_processing(self, valuations):
        return valuations


class SpatialBundle(RequestSelectionStrategyBundle):
    """
    Gansterer & Hartl (2016) refer to this one as 'cluster'
    """

    def evaluate_bundle(self, instance: it.CAHDInstance, carrier: core_module.carrier.Carrier, bundle: Bundle):
        """
        the sum of travel distances of all pairs of requests in this cluster, where the travel distance of a request
        pair is defined as the sum of their asymmetric distances
        """
        return bundle._travel_duration_matrix.sum()


class TemporalBundle(RequestSelectionStrategyBundle):
    def evaluate_bundle(self, instance: it.CAHDInstance, carrier: core_module.carrier.Carrier, bundle: Bundle):
        """
        sum of pairwise time window distances of requests in this cluster.
        distance between TW a and TW b is abs(a.center - b.center)
        """
        return abs(bundle._time_window_distance_matrix).sum()


class TemporalRangeBundle(RequestSelectionStrategyBundle):

    def evaluate_bundle(self, instance: it.CAHDInstance, carrier: core_module.carrier.Carrier, bundle: Bundle):
        """
        the min-max range of the delivery time windows of all requests inside the cluster
        """

        bundle_tw_open = [instance.tw_open[v] for v in bundle.vertices]
        bundle_tw_close = [instance.tw_close[v] for v in bundle.vertices]
        min_open: dt.datetime = min(bundle_tw_open)
        max_close: dt.datetime = max(bundle_tw_close)
        # negative value: low temporal range means high valuation
        evaluation = (max_close - min_open).total_seconds()
        return evaluation


class LinearCombination(RequestSelectionStrategyBundle):
    def __init__(self,
                 num_of_submitted_requests: int,
                 components: Sequence[RequestSelectionStrategyBundle],
                 weights: Sequence[float] = None,
                 assert_sum_1: bool = True):
        super().__init__(num_of_submitted_requests)
        assert len(components) == len(weights)
        self.components = components
        if weights is None:
            weights = [Fraction(1, len(components)) for _ in components]
        if assert_sum_1:
            assert sum(weights) == 1
        self.weights = weights

        self.name = self.__class__.__name__ + '_' + '_'.join(str(w) + '*' + c.name for w, c in zip(weights, components))

    def evaluate_bundle(self, instance: it.CAHDInstance, carrier: core_module.carrier.Carrier, bundle: Bundle):
        return [comp.evaluate_bundle(instance, carrier, bundle) for comp in self.components]

    def valuations_post_processing(self, valuations):
        """
        Normalize/standardize the different components and return their weighted sum

        :param valuations:
        :return:
        """
        normalized_components = []

        # min-max feature scaling
        for component_series in zip(*valuations):
            min_comp, max_comp = min(component_series), max(component_series)
            normalized_components.append([(x - min_comp) / (max_comp - min_comp) for x in component_series])

        # compute the weighted sum
        valuations = [sum(np.array(self.weights) * comp) for comp in zip(*normalized_components)]
        return valuations


class SpatioTemporalBundle(RequestSelectionStrategyBundle):
    """
    The evaluation of a bundle is an aggregation (e.g., sum or mean) of pairwise spatial and temporal distances of all
    requests in the bundle

    """

    def preprocessing(self, instance: it.CAHDInstance, solution: slt.CAHDSolution):
        # since values must be min-max normalized, I need min and max travel_durations & tw_distances per carrier
        self.carrier_min_travel_duration = []
        self.carrier_max_travel_duration = []
        self.carrier_min_tw_distance = []
        self.carrier_max_tw_distance = []
        for carrier_id, carrier in enumerate(solution.carriers):
            mask = np.ones((len(carrier.accepted_requests), len(carrier.accepted_requests)), dtype=bool)
            np.fill_diagonal(mask, 0)

            dur_m = instance._travel_duration_matrix[np.ix_(carrier.accepted_requests, carrier.accepted_requests)]
            self.carrier_min_travel_duration.append(dur_m[mask].min())
            self.carrier_max_travel_duration.append(dur_m[mask].max())

            carrier_vertices = [instance.vertex_from_request(r) for r in carrier.accepted_requests]
            tw_m = np.array([[abs(instance.time_window(v0).center - instance.time_window(v1).center)
                              for v1 in carrier_vertices] for v0 in carrier_vertices])
            self.carrier_min_tw_distance.append(tw_m[mask].min())
            self.carrier_max_tw_distance.append(tw_m[mask].max())
        return

    def evaluate_bundle(self, instance: it.CAHDInstance, carrier: core_module.carrier.Carrier, bundle: Bundle):
        """
        a bundle is good if requests are close together in terms of both travel duration AND in terms of time windows.
        Bundle valuation is the sum of pairwise (a) travel durations and (b) time window distances, were these two
        dimensions are normalized.


        :param instance:
        :param carrier:
        :param bundle:
        :return:
        """
        travel_durations = []
        tw_distances = []
        for v0, v1 in itertools.combinations(bundle.vertices, 2):
            # normalized travel duration
            duration = instance.travel_duration([v0], [v1])
            travel_durations.append((duration - self.carrier_min_travel_duration[carrier.id_]) / (
                    self.carrier_max_travel_duration[carrier.id_] - self.carrier_min_travel_duration[carrier.id_]))

            # normalized tw distance
            tw_dist = abs(instance.time_window(v0).center - instance.time_window(v1).center)
            if tw_dist != dt.timedelta(0):
                tw_distances.append((tw_dist - self.carrier_min_tw_distance[carrier.id_]) / (
                        self.carrier_max_tw_distance[carrier.id_] - self.carrier_min_tw_distance[carrier.id_]))
            else:
                tw_distances.append(0)

        evaluation = self.aggregate(travel_durations, tw_distances, bundle)
        return evaluation

    def aggregate(self, normalized_travel_durations, normalized_tw_distances, bundle: Bundle):
        return sum(normalized_travel_durations) + sum(normalized_tw_distances)


class LosSchulteBundle(RequestSelectionStrategyBundle):
    """
    Selects requests based on their combined evaluation of the bundle evaluation measure by [1] Los, J., Schulte, F.,
    Gansterer, M., Hartl, R. F., Spaan, M. T. J., & Negenborn, R. R. (2020). Decentralized combinatorial auctions for
    dynamic and large-scale collaborative vehicle routing.
    """

    def evaluate_bundle(self, instance: it.CAHDInstance, carrier: core_module.carrier.Carrier, bundle: Bundle):
        # must invert since RequestSelectionBehaviorBundle searches for the maximum valuation and request
        return bundle.LS_spatio_temporal_cohesion


class AvgDepotDurations(RequestSelectionStrategyBundle):
    """
    WARNING: Theoretically, this must have the same outcome as the individual DepotDurations
    Consider a bundle's average of the measure DepotDurations, where DepotDurations is defined as the minimum duration
    to any foreign depot divided by the duration to the own depot: min_dur_to_foreign_depot / dur_to_own_depot
    """

    def evaluate_bundle(self, instance: it.CAHDInstance, carrier: core_module.carrier.Carrier, bundle: Bundle):
        raise UserWarning('WARNING: Theoretically, this must have the same outcome as the individual DepotDurations')
        # [i] duration to the depot of one of the collaborating carriers
        min_dur_to_foreign_depot = []
        dur_to_own_depot = []
        foreign_depots = list(range(instance.num_carriers))
        foreign_depots.pop(carrier.depot_vertex)
        for request in bundle:
            # [i] travel duration to closest foreign depot
            min_foreign = dt.timedelta.max
            vertex = instance.vertex_from_request(request)
            for depot in foreign_depots:
                dur = sum(
                    (instance.travel_duration([depot], [vertex]), instance.travel_duration([vertex], [depot])),
                    start=dt.timedelta(0))
                if dur < min_foreign:
                    min_foreign = dur
            # [ii] travel duration to own depot
            own = instance.travel_duration([carrier.depot_vertex, vertex], [vertex, carrier.depot_vertex]) / 2
            min_dur_to_foreign_depot.append(min_foreign)
            dur_to_own_depot.append(own)
        evaluation = [a / b for a, b in zip(min_dur_to_foreign_depot, dur_to_own_depot)]
        evaluation = sum(evaluation) / len(evaluation)
        return evaluation
