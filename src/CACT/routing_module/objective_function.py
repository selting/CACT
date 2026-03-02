from abc import abstractmethod
import datetime
from core_module.instance import CAHDInstance
from core_module.request import Request
from core_module.tour import Tour


class ObjectiveFunction:
    def __init__(self):
        self.best = None
        self.worst = None
        self.higher_is_better: bool = None
        pass

    def __repr__(self):
        return f"Objective({self.__class__.__name__})"

    @abstractmethod
    def evaluate(self, tour: Tour):
        return self(tour)

    @abstractmethod
    def insertion_criterion(
        self, instance: CAHDInstance, tour: Tour, pos: int, request: Request
    ) -> float:
        pass

    def is_better(self, a, b):
        """whether a is better than b. This depends on whether the objective follows a higher-is-better paradigm or not.

        Args:
            a (Tour): _description_
            b (Tour): _description_
        """
        if self.higher_is_better:
            return a > b
        else:
            return a < b


class MinDistance(ObjectiveFunction):
    def __init__(self):
        self.best = 0
        self.worst = float("inf")
        self.higher_is_better = False
        pass

    def evaluate(self, tour: Tour):
        return tour.sum_travel_distance

    def insertion_criterion(
        self, instance: CAHDInstance, tour: Tour, pos: int, request: Request
    ) -> float:
        delta = tour.insert_distance_delta(instance, [pos], [request])
        return delta


class MinDuration(ObjectiveFunction):
    def __init__(self):
        self.best = datetime.timedelta(0)
        self.worst = datetime.timedelta.max
        self.higher_is_better = False
        pass

    def evaluate(self, tour):
        return tour.sum_travel_duration

    def insertion_criterion(self, instance, tour, pos, request):
        delta = tour.insert_duration_delta
        return delta