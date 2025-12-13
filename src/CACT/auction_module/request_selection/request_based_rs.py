import random
from abc import ABC, abstractmethod
from fractions import Fraction
from statistics import mean
from typing import Sequence, Callable

import numpy as np

from auction_module.request_selection.request_selection import RequestSelectionStrategy
from core_module.carrier import Carrier
from core_module.instance import CAHDInstance
from core_module.request import Request
from core_module.solution import CAHDSolution
from core_module.tour import Tour
from tw_management_module.time_window import TimeWindow


class RequestSelectionStrategyIndividual(RequestSelectionStrategy, ABC):
    """
    select (for each carrier) a set of bundles based on their individual evaluation of some quality criterion
    """

    def __call__(self, instance: CAHDInstance, tours: list[Tour], k: int) -> [Request]:
        """
        select a set of requests based on the concrete selection behavior. will retract the requests from the carrier
        and return. Low evaluation values are better

        :return: the Assignment of requests to carriers
        """
        valuations = dict()
        for tour in tours:
            for request in tour.requests:
                valuations[request] = self._evaluate_request(instance, tours, request)

        valuations = self.post_processing(valuations)

        # pick the LOWEST k evaluated requests (from ascending order)
        # a) without random tie breaking
        # selected = sorted(carrier.accepted_requests, key=)[:k]
        # b) WITH random tie breaking
        selected = sorted(valuations, key=lambda x: valuations[x])[:k]

        return selected

    @abstractmethod
    def _evaluate_request(self, instance: CAHDInstance, tours: list[Tour], request: Request) -> float:
        """compute the valuation of the given request for the carrier"""
        pass

    def post_processing(self, valuations):
        return valuations


class Random(RequestSelectionStrategyIndividual):
    """
    returns a random selection of unrouted requests
    """

    def _evaluate_request(self, instance: CAHDInstance, tours: list[Tour], request: Request) -> float:
        return random.random()


class MarginalTravelDurationProxy(RequestSelectionStrategyIndividual):
    """
    RS based on the marginal duration induced to the tour by the currently regarded request j. Requests with high
    marginal duration are likely to be submitted
    Marginal duration is estimated by: duration(i, k) - duration(i, j) - duration(j, k). Thus: the lower the value, the
    longer the detour
    """

    def _evaluate_request(self, instance: CAHDInstance, tours: list[Tour], request: Request) -> float:
        for tour in tours:
            if request in tour.requests:
                tour_of_request = tour
                break
        pos = tour_of_request.vertex_pos[request]
        evaluation = tour_of_request.pop_duration_delta(instance, [pos]).total_seconds()
        return evaluation


class RequestTime(RequestSelectionStrategyIndividual):
    def _evaluate_request(self, instance: CAHDInstance, tours: list[Tour], request: Request) -> float:
        return request.disclosure_time


class DurationToOwnDepot(RequestSelectionStrategyIndividual):
    """Release requests that are far away from the own depot. (will return the negative of the duration)"""

    def __init__(self, asym_dur_aggr: Callable):
        """

        :param asym_dur_aggr:
        """
        super().__init__()
        self.asym_dur_aggr = asym_dur_aggr
        self.name = asym_dur_aggr.__name__.capitalize() + self.__class__.__name__

    def _evaluate_request(self, instance: CAHDInstance, tours: list[Tour], request: Request) -> float:
        # lower is better, thus, take the negative
        depot = tours[0].routing_sequence[0]
        return - self.asym_dur_aggr((instance.travel_duration([depot], [request]).total_seconds(),
                                     instance.travel_duration([request], [depot]).total_seconds()))


class DurationToForeignDepot(RequestSelectionStrategyIndividual):
    def __init__(self, num_submitted_requests: int, depot_dur_aggr: Callable, asym_dur_aggr: Callable):
        """

        :param num_submitted_requests:
        :param depot_dur_aggr:  How should all the durations to the foreign depots be aggregated? (e.g., min, max, mean)
        :param asym_dur_aggr: by which aggregation function (e.g. min, max, mean) shall the two duration measures from
        and to a foreign depot be aggregated?
        """
        raise NotImplementedError('Is not yet compatible with the new request selection interface')
        super().__init__(num_submitted_requests)
        self.depot_dur_aggr = depot_dur_aggr
        self.asym_dur_aggr = asym_dur_aggr
        self.name = depot_dur_aggr.__name__.capitalize() + asym_dur_aggr.__name__.capitalize() + self.__class__.__name__

    def _evaluate_request(self, instance: CAHDInstance, tours: list[Tour], request: Request) -> float:
        # TODO it is not known to the function which ware the foreign and which is the own depot
        foreign_depots = [carrier.depot for carrier in solution.carriers if carrier != carrier]
        return self.depot_dur_aggr(self.asym_dur_aggr((instance.travel_duration([depot], [request]).total_seconds(),
                                                       instance.travel_duration([request], [depot]).total_seconds()))
                                   for depot in foreign_depots)


class ForwardTimeSlack(RequestSelectionStrategyIndividual):

    def _evaluate_request(self, instance: CAHDInstance, tours: list[Tour], request: Request) -> float:
        for tour in tours:
            if request in tour.requests:
                tour_of_request = tour
                break
        pos = tour_of_request.vertex_pos[request]
        return t.max_shift_sequence[pos]


class LinearCombination(RequestSelectionStrategyIndividual):
    """
    computes a request's valuation as the weighted sum of its value quantities for some simpler request quality
    functions. before computing the weighted sum, all components are standardized using mean and standard deviation.
    """

    def __init__(self,
                 num_submitted_requests: int,
                 components: Sequence[RequestSelectionStrategyIndividual],
                 weights: Sequence[float] = None,
                 assert_sum_1: bool = True):
        raise NotImplementedError('Is not yet compatible with the new request selection interface')

        super().__init__(num_submitted_requests)
        assert len(components) == len(weights)
        self.components = components
        if weights is None:
            weights = [Fraction(1, len(components)) for _ in components]
        if assert_sum_1:
            assert sum(weights) == 1
        self.weights = weights

        self.name = self.__class__.__name__ + '_' + '_'.join(str(w) + '*' + c.name for w, c in zip(weights, components))

    def _evaluate_request(self, instance: CAHDInstance, solution: CAHDSolution, carrier: Carrier, request: Request):
        return [comp.evaluate_request(instance, solution, carrier, request) for comp in self.components]

    def post_processing(self, valuations: dict[Request, float]):
        normalized_components = []

        '''
        # z-score standardization of the valuation components
        for component_series in zip(*valuations):
            mean = sum(component_series) / len(component_series)
            std = sqrt(sum([(x - mean) ** 2 for x in component_series]) / len(component_series))
            standardized_components.append(((x - mean) / std for x in component_series))
        '''
        raise NotImplementedError('needs update')
        # min-max feature scaling
        for component_series in zip(*valuations):
            min_comp, max_comp = min(component_series), max(component_series)
            if min_comp == max_comp:
                normalized_components.append([1 for _ in component_series])
            else:
                normalized_components.append([(x - min_comp) / (max_comp - min_comp) for x in component_series])

        # compute the weighted sum of the standardized components
        valuations = [sum(np.array(self.weights) * comp) for comp in zip(*normalized_components)]
        return valuations


class DepotDurations(RequestSelectionStrategyIndividual):
    """
    considers min_dur_to_foreign_depot and dur_to_own_depot. Does NOT consider marginal profit!
    evaluates requests as min_dur_to_foreign_depot/dur_to_own_depot
    """

    def _evaluate_request(self, instance: CAHDInstance, tours: list[Tour], request: Request) -> float:
        raise NotImplementedError('Is not yet compatible with the new request selection interface')

        # [i] duration to the depot of one of the collaborating carriers
        min_dur_to_foreign_depot = DurationToForeignDepot(None, min, mean).evaluate_request(instance, solution,
                                                                                            carrier, request)

        # [ii] duration to the carrier's own depot
        # duration as the average of the asymmetric durations between depot and delivery vertex
        dur_to_own_depot = instance.travel_duration([carrier.depot, request], [request, carrier.depot]) / 2

        return min_dur_to_foreign_depot / dur_to_own_depot.total_seconds()


class TimeWindowFillLevel(RequestSelectionStrategyIndividual):
    """Release requests from those time windows that only few customers have chosen"""

    def _evaluate_request(self, instance: CAHDInstance, tours: list[Tour], request: Request) -> float:
        tw = TimeWindow(request.tw_open, request.tw_close)
        num = 0
        all_requests = []
        for tour in tours:
            all_requests.extend(tour.requests)
        for r in all_requests:  # for r in solution.tour_of_request(request).requests:
            if (r.tw_open, r.tw_close) == tw:
                num += 1
        return num
