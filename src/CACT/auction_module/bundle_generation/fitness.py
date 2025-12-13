import random
from abc import ABC, abstractmethod
from typing import Union, Sequence

from auction_module.bundle_generation.assignment_based.assignment import Assignment
from auction_module.bundle_generation.bundle_based.bundle import Bundle
from auction_module.bundle_generation.partition_based.partition import Partition
from core_module.instance import CAHDInstance
from core_module.solution import CAHDSolution


class Fitness(ABC):
    """
    Abstract class for fitness functions. A fitness function can be defined for bundles, for partitions or for
    assignments.
    In general, it holds that the higher the fitness value, the better the individual.
    """

    def __init__(self):
        self.name = self.__class__.__name__

    def __repr__(self):
        return self.name

    def __call__(self, instance: CAHDInstance, solution: CAHDSolution,
                 individual: Union[Bundle, Partition, Assignment]) -> float:
        return self.evaluate(instance, solution, individual)

    @abstractmethod
    def evaluate(self, instance: CAHDInstance, solution: CAHDSolution,
                 individual: Union[Bundle, Partition, Assignment]) -> float:
        pass


class FitnessCarrierAware(Fitness):
    """returns a list of fitness values, one for each carrier"""

    @abstractmethod
    def evaluate(self, instance: CAHDInstance, solution: CAHDSolution, assignment: Assignment) -> Sequence[float]:
        pass


class Random(Fitness):
    def evaluate(self, instance: CAHDInstance, solution: CAHDSolution, assignment: Assignment) -> float:
        return random.random()


class RandomCA(FitnessCarrierAware):
    def evaluate(self, instance: CAHDInstance, solution: CAHDSolution, assignment: Assignment) -> Sequence[float]:
        return [random.random() for _ in range(instance.num_carriers)]
