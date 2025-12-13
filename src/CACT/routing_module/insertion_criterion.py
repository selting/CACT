import abc

from core_module.instance import CAHDInstance
from core_module.request import Request
from core_module.tour import Tour
from utility_module.parameterized_class import ParameterizedClass


class InsertionCriterion(ParameterizedClass):

    def __repr__(self):
        return self.__class__.__name__

    @abc.abstractmethod
    def __call__(self, instance: CAHDInstance, tour: Tour, pos: int, request: Request) -> float:
        pass


class MinDuration(InsertionCriterion):
    def __call__(self, instance: CAHDInstance, tour: Tour, pos: int, request: Request) -> float:
        delta = tour.insert_duration_delta(instance, [pos], [request]).total_seconds()
        return delta
