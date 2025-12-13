import random
from abc import ABC, abstractmethod

import core_module.carrier
from core_module import instance as it, solution as slt
from core_module.request import Request


# TODO: many of these need an overhaul, their exact implementation is not tuned!
class CustomerAcceptanceMeasure(ABC):
    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def evaluate(self, instance: it.CAHDInstance, carrier: core_module.carrier.Carrier, request: Request) -> bool:
        """
        return True if the request shall be accepted and False if it should be rejected based on the class approach
        to assessing/evaluating requests

        :param instance:
        :param carrier:
        :param request:
        :return:
        """
        pass

    @abstractmethod
    def evaluate_central(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, request: Request) -> bool:
        pass


class Accept(CustomerAcceptanceMeasure):
    """
    Accepts incoming requests as long as feasible in First Come First Served order
    """

    def evaluate(self, instance: it.CAHDInstance, carrier: core_module.carrier.Carrier, request: Request) -> bool:
        return True

    def evaluate_central(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, request: Request) -> bool:
        return True


class Reject(CustomerAcceptanceMeasure):
    """
    Rejects all requests
    """

    def evaluate(self, instance: it.CAHDInstance, carrier: core_module.carrier.Carrier, request: Request) -> bool:
        return False

    def evaluate_central(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, request: Request) -> bool:
        return False


class AcceptCloseToDepot(CustomerAcceptanceMeasure):
    """
    Only accept requests that are within a certain 'radius' around the depot are accepted.
    Let v be the (approximated) duration from the carrier depot to the city border and let w be the duration from the
    carrier depot to the regarded request. Whether a request is accepted is defined as: w/v <= threshold
    """

    def __init__(self, radius: float):
        super().__init__()
        assert 0 <= radius <= 1, 'Threshold must be in the interval [0, 1]'
        self.radius = radius
        self.name = self.__class__.__name__ + f'r={radius}'

    def evaluate(self, instance: it.CAHDInstance, carrier: core_module.carrier.Carrier, request: Request) -> bool:
        vertex = instance.vertex_from_request(request)
        attractiveness = instance.travel_duration([carrier.depot_vertex, vertex], [vertex, carrier.depot_vertex]) / 2
        attractiveness = attractiveness / instance._travel_duration_matrix[
            carrier.depot_vertex].max()  # [0, 1] normalize
        if attractiveness <= self.radius:
            return True
        else:
            return False


class AcceptCloseToCompetitor(CustomerAcceptanceMeasure):
    def __init__(self, closeness: float):
        super().__init__()
        assert 0 <= closeness <= 1
        self.closeness = closeness
        self.name = self.__class__.__name__ + str(int(closeness * 100))

    def evaluate(self, instance: it.CAHDInstance, carrier: core_module.carrier.Carrier, request: Request) -> bool:
        vertex = instance.vertex_from_request(request)
        carrier_dur_sum = instance.travel_duration([carrier.depot_vertex, vertex],
                                                   [vertex, carrier.depot_vertex])
        for competitor_id in range(instance.num_carriers):
            if competitor_id == carrier.id_:
                continue
            competitor_dur_sum = instance.travel_duration([competitor_id, vertex],
                                                          [vertex, competitor_id])
            # self.closeness = 0.5 -> the request is at least as close to the competitor as it is to the carrier
            if competitor_dur_sum / (competitor_dur_sum + carrier_dur_sum) <= self.closeness:
                return True
        return False


class AcceptCloseToAny(CustomerAcceptanceMeasure):
    """
    accept requests that are close to the own or close to a competitor's depot.
    """

    def __init__(self, depot_closeness: float, competitor_closeness: float):
        super().__init__()
        assert 0 <= depot_closeness <= 1
        assert 0 <= competitor_closeness <= 1
        self.depot_closeness = depot_closeness
        self.competitor_closeness = competitor_closeness
        self.name = self.__class__.__name__ + f'_d={depot_closeness}_c={competitor_closeness}'

    def evaluate(self, instance: it.CAHDInstance, carrier: core_module.carrier.Carrier, request: Request) -> bool:
        vertex = instance.vertex_from_request(request)

        # depot
        depot_attrac = instance.travel_duration([carrier.depot_vertex, vertex], [vertex, carrier.depot_vertex]) / 2
        depot_attrac = depot_attrac / instance._travel_duration_matrix[carrier.depot_vertex].max()
        if depot_attrac <= self.depot_closeness:
            return True

        # competitor
        for competitor_id in range(instance.num_carriers):
            if competitor_id == carrier.id_:
                continue
            competitor_attrac = instance.travel_duration([competitor_id, vertex], [vertex, competitor_id])
            competitor_attrac = competitor_attrac / instance._travel_duration_matrix[competitor_id].max()
            if competitor_attrac <= self.competitor_closeness:
                return True

        return False


class DynamicAcceptCloseToAlreadyAccepted(CustomerAcceptanceMeasure):
    """
    accept only customers that are in the vicinity of an already accepted customer. the threshold for the maximum
    allowed additional travel duration is dependent on the current utilization level
    """

    def __init__(self, n, m):
        super().__init__()
        self.n = n
        self.m = m
        self.name = self.__class__.__name__ + f'_n={n}_m={m}'

    def evaluate(self, instance: it.CAHDInstance, carrier: core_module.carrier.Carrier, request: Request) -> bool:
        vertex = instance.vertex_from_request(request)
        accepted_vertices = [instance.vertex_from_request(r) for r in carrier.accepted_requests]
        if len(accepted_vertices) == 0:
            return True
        min_dur_neighbor = instance._travel_duration_matrix[vertex][accepted_vertices].min()
        min_dur_neighbor = min_dur_neighbor / instance._travel_duration_matrix[
            carrier.depot_vertex].max()  # FIXME does not consider different overlaps levels
        threshold = 1 - carrier.utilization() ** self.n + self.m
        if min_dur_neighbor <= threshold:
            return True
        else:
            return False


class DynamicAcceptCloseToDepot(CustomerAcceptanceMeasure):
    """
    Accept far away customers only at the very beginning or the very end.
    Decrease and increase the radius
    """

    def __init__(self, a: float, b: float, c: float):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.name = self.__class__.__name__ + f'a={a}_b={b}_c={c}'

    def evaluate(self, instance: it.CAHDInstance, carrier: core_module.carrier.Carrier, request: Request) -> bool:
        vertex = instance.vertex_from_request(request)
        attractiveness = instance.travel_duration([carrier.depot_vertex, vertex], [vertex, carrier.depot_vertex]) / 2
        attractiveness = attractiveness / instance._travel_duration_matrix[carrier.depot_vertex].max()
        min_threshold = -(self.a * (carrier.utilization() - self.b) ** 2) + self.c
        if attractiveness >= min_threshold:
            return True
        else:
            return False


class DynamicAttractivenessAndUtilization(CustomerAcceptanceMeasure):
    def __init__(self, a, b, c, d):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.name = self.__class__.__name__ + f'_a={a}_b={b}_c={c}_d={d}'

    def evaluate(self, instance: it.CAHDInstance, carrier: core_module.carrier.Carrier, request: Request) -> bool:
        vertex = instance.vertex_from_request(request)
        utilization = carrier.utilization()
        # FIXME find a proper measure, e.g. average time_shift over all late TWs (assuming that these are most likely to be selected)
        attractiveness = instance.travel_duration([carrier.depot_vertex], [vertex])
        attractiveness = attractiveness / instance._travel_duration_matrix[
            carrier.depot_vertex].max()  # [0, 1] normalize

        acceptance_probability = (self.a * (utilization - self.b) ** 2) + (self.c * attractiveness ** 2) + self.d
        r = random.random()
        if acceptance_probability > r:
            return True
        else:
            return False
