from abc import ABC, abstractmethod

from auction_module.bundle_generation.assignment_based.assignment import Assignment
from auction_module.bundle_generation.fitness import Fitness
from core_module import instance as it, solution as slt
from core_module.request import Request


class BundleGenerationBehavior(ABC):
    """
    Generates auction bundles based on partitioning the auction request pool. This guarantees a feasible solution
    for the Winner Determination Problem which cannot (easily) be guaranteed if bundles are generated without
    considering that the WDP requires a partitioning of the auction request pool (see bundle_based_bg.py)
    """

    def __init__(self, fitness: Fitness):
        self.fitness = fitness
        self.name = self.__class__.__name__

    def __repr__(self):
        return self.name

    @abstractmethod
    def execute(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, auction_request_pool: tuple[Request],
                num_bidding_jobs: int, original_assignment: Assignment) -> list[list[tuple[int]]]:
        """

        :param instance:
        :param solution:
        :param auction_request_pool:
        :param num_bidding_jobs: number of bidding jobs to generate per carrier
        :param original_assignment:
        :return:
        """
        pass
