from abc import abstractmethod
from typing import Callable

from auction_module.bundle_generation.assignment_based.assignment import Assignment
from auction_module.bundle_generation.fitness import Fitness
from core_module.instance import CAHDInstance
from core_module.solution import CAHDSolution


class AssignmentFitness(Fitness):
    @abstractmethod
    def evaluate(
        self, instance: CAHDInstance, solution: CAHDSolution, assignment: Assignment
    ) -> float:
        pass


# class TrueBidsAndCAP(AssignmentFitness):
#     def __init__(self, bidding_behavior: BiddingBehavior):
#         super().__init__()
#         self.bidding_behavior = bidding_behavior
#
#     def evaluate(self, instance: CAHDInstance, solution: CAHDSolution, assignment: Assignment) -> float:
#         bidding_jobs_df = assignments_to_bidding_job_df(instance, solution, [assignment], None)
#         bids_matrix = self.bidding_behavior.execute_bidding(instance, solution, bidding_jobs_df, True)
#         bids_matrix = bids_matrix.applymap(lambda x: x.total_seconds() if isinstance(x, dt.timedelta) else x)
#         z = bids_matrix.sum().sum()
#         # smaller is better, so we return the negative value
#         return -z


# ______________________________________________________________________________________________________________________
# AGGREGATE ASSIGNMENT FITNESS
# ______________________________________________________________________________________________________________________


class AggregateAssignmentFitness(AssignmentFitness):
    def __init__(self, aggr: Callable):
        super().__init__()
        self.aggr = aggr
        self.name = aggr.__name__.capitalize() + "_" + self.__class__.__name__

    @abstractmethod
    def evaluate(
        self, instance: CAHDInstance, solution: CAHDSolution, assignment: Assignment
    ) -> float:
        pass


class CentroidSSE(AggregateAssignmentFitness):
    def evaluate(
        self, instance: CAHDInstance, solution: CAHDSolution, assignment: Assignment
    ) -> float:
        sse = self.aggr(
            [
                bundle.sum_of_squared_centroid_distance
                for bundle, carrier in assignment.bundle_to_carrier().items()
            ]
        )
        # smaller is better, so we return the negative value
        return -sse


class DepotSSE(AggregateAssignmentFitness):
    def evaluate(
        self, instance: CAHDInstance, solution: CAHDSolution, assignment: Assignment
    ) -> float:
        sse = self.aggr(
            [
                bundle.sum_of_squared_duration_to_carrier_depot(carrier)
                for bundle, carrier in assignment.bundle_to_carrier().items()
            ]
        )
        # smaller is better, so we return the negative value
        return -sse


class Tsp(AggregateAssignmentFitness):
    def evaluate(
        self, instance: CAHDInstance, solution: CAHDSolution, assignment: Assignment
    ) -> float:
        res = []
        for bundle, carrier in assignment.bundle_to_carrier().items():
            res.append(bundle.tour_sum_travel_duration)
        aggr_res = self.aggr(res)
        # smaller is better, so we return the negative value
        return -aggr_res


# ______________________________________________________________________________________________________________________
