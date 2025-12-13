import random
from abc import abstractmethod
from typing import Sequence, Any, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import core_module.carrier
from auction_module.bundle_generation.assignment_based.assignment import Assignment
from auction_module.bundle_generation.bundle_based.bundle import Bundle
from auction_module.bundle_generation.bundle_generation import BundleGenerationBehavior
from auction_module.bundle_generation.partition_based.partition import Partition
from auction_module.bundling_and_bidding.type_defs import QueriesType
from core_module import instance as it, solution as slt
from core_module.request import Request
from utility_module.datetime_handling import total_seconds_vectorized
from utility_module.utils import debugger_is_active


class NumBiddingTasksFromAssignments(BundleGenerationBehavior):
    """
    Generate a pool of bundles that has a limited, predefined number of bundles in it.
    Bundle selection happens by evaluating different partitions of the auction request pool and keeping a partition's
    bundles if the partition's evaluation was sufficiently good.

    Note: the size of the resulting auction pool may differ slightly from the exact given number, since partitions
    are always preserved. Therefore, whenever a partition's evaluation is sufficient, all its bundles will be kept,
    even if this means that the given bundle pool size is slightly exceeded.
    """

    @abstractmethod
    def execute(
        self,
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        auction_request_pool: tuple[Request],
        num_bidding_jobs: int,
        original_assignment: Assignment,
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def generate_assignment(
        self,
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        auction_request_pool: tuple[Bundle],
    ):
        pass


class BestOfManyMaxKAssignments(NumBiddingTasksFromAssignments):
    def __init__(
        self, fitness, many: int
    ):  # TODO why does this not have max_k as parameter?
        super().__init__(fitness)
        self.many = many
        self.name = self.__class__.__name__.replace("Many", str(many))

    def execute(
        self,
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        auction_request_pool: tuple[Request],
        num_bidding_jobs: int,
        original_assignment: Assignment,
    ) -> Tuple[QueriesType, Optional[Sequence[dict[Partition, float]]]]:
        # if the number of all possible bundles is lower than then number of required bidding jobs -> generate all bundles
        if 2 ** len(auction_request_pool) - 1 < num_bidding_jobs:
            raise ValueError(
                f"Number of bidding jobs ({num_bidding_jobs}) exceeds the number of possible bundles "
                f"({2 ** len(auction_request_pool) - 1})"
            )

        assignments = []
        fitness = []
        with tqdm(
            total=self.many,
            desc=f"Generating assignments candidates",
            disable=not debugger_is_active(),
        ) as pbar:
            while len(assignments) < self.many:
                ass = self.generate_assignment(instance, solution, auction_request_pool)
                if ass not in assignments:
                    assignments.append(ass)
                    fitness.append(self.fitness(instance, solution, ass))
                    pbar.update()

        combined = sorted(
            zip(assignments, fitness), key=lambda x: x[1], reverse=True
        )  # max fitness at the top
        assignments_sorted, fitness_sorted = zip(*combined)
        # assignments_sorted = (original_assignment,) + assignments_sorted
        # bidding_jobs_df = assignments_to_bidding_job_df(instance, solution, assignments_sorted, num_bidding_jobs)
        # return bidding_jobs_df

        bidding_jobs = [
            [
                original_assignment.carrier_to_bundle()[carrier_idx].bitstring,
            ]
            for carrier_idx in range(instance.num_carriers)
        ]
        ass: Assignment
        for ass in assignments_sorted:
            for carrier_idx in range(instance.num_carriers):
                bundle = ass.carrier_to_bundle()[carrier_idx].bitstring
                if any(bundle) and bundle not in bidding_jobs[carrier_idx]:
                    bidding_jobs[carrier_idx].append(bundle)
            if all(len(x) >= num_bidding_jobs for x in bidding_jobs):
                break
        return bidding_jobs, [
            Partition.from_assignment(ass).normalize() for ass in assignments
        ]

    @abstractmethod
    def generate_assignment(
        self,
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        auction_request_pool: tuple[Bundle],
    ):
        pass


class BestOfManyRandomMaxKAssignments(BestOfManyMaxKAssignments):
    def generate_assignment(
        self,
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        auction_request_pool: tuple[Bundle],
    ):
        return Assignment.random(solution.carriers, auction_request_pool)


class TravelDurationRouletteWheelAssignments(BestOfManyRandomMaxKAssignments):
    """
    assigns each request to a random carrier based on a probability that is proportional to the travel duration of
    the carrier's depot to the request location.
    """

    def generate_assignment(
        self,
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        auction_request_pool: tuple[Bundle],
    ):
        assignment = Assignment(solution.carriers)

        request_vertices = [
            instance.vertex_from_request(r) for r in auction_request_pool
        ]
        depot_vertices = [c.depot_vertex for c in solution.carriers]
        travel_duration = instance._travel_duration_matrix[request_vertices][
            :, depot_vertices
        ]
        probabilities = total_seconds_vectorized(travel_duration)
        probabilities = 1 / probabilities
        probabilities = probabilities / probabilities.sum(1, keepdims=True)
        rng = np.random.default_rng()
        for i in range(len(auction_request_pool)):
            selected_carrier_index = rng.choice(
                instance.num_carriers, p=probabilities[i]
            )
            assignment[auction_request_pool[i]] = selected_carrier_index
        return assignment

        # previous version - did not take advantage of numpy vectorization
        # roulette wheel selection
        # for request in auction_request_pool:
        #     fitness_values = []
        #     for carrier in solution.carriers:
        #         duration = instance.travel_duration([instance.vertex_from_request(request)], [carrier.depot_vertex])
        #         fitness = 1 / duration.total_seconds()
        #         fitness_values.append(fitness)
        #
        #     probabilities = np.array(fitness_values) / sum(fitness_values)
        #     selected_idx = np.random.choice(instance.num_carriers, p=probabilities)
        #     assignment[request] = selected_idx
        # return assignment


class MiniClustersRandomAssignment(BestOfManyRandomMaxKAssignments):
    """
    Generates mini-clusters of 2-3 requests that are within proximity and assigns each cluster to a random carrier.
    """

    def generate_assignment(
        self,
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        auction_request_pool: tuple[Bundle],
    ):
        assignment = Assignment(solution.carriers)
        # generate mini-clusters
        mini_clusters = []
        available_requests = set(auction_request_pool)
        nn = instance.nearest_neighbors_durations
        while available_requests:
            # select a random request
            request = available_requests.pop()
            cluster = [request]
            # select a random number of requests to add to the cluster
            k = random.choice([2, 3])
            knn = np.argpartition(nn[request], k)[:k]  # not sorted!
            cluster.extend(knn)
            available_requests.difference_update(cluster)
            mini_clusters.append(cluster)

        # assign each cluster to a random carrier
        for cluster in mini_clusters:
            selected_idx = random.choice(range(instance.num_carriers))
            for request in cluster:
                assignment[request] = selected_idx
        return assignment


def assignments_to_bidding_job_df(
    instance,
    solution: slt.CAHDSolution,
    sorted_assignments: Sequence[Assignment],
    max_num_bidding_jobs: int = None,
):
    """
    Creates a bidding job DataFrame from a sequence of assignments. Iterates through the assignments and extracts the
    jobs from each assignment until max_num_bidding_jobs have been collected. If max_num_bidding_jobs is None, all
    assignments are used.

    :param instance: CAHDInstance
    :param solution: CAHDSolution
    :param sorted_assignments: Iterable[Assignment], should be sorted by fitness!
    :param max_num_bidding_jobs: optional integer, specifying the maximum number of bidding jobs to be created. Useful
    if there is a limit on the number of bidding jobs that should be created. If None, all assignments are used.

    """
    if max_num_bidding_jobs is None:
        max_num_bidding_jobs = len(sorted_assignments) * len(solution.carriers)
    records: dict[Bundle, dict[Any, Any]] = dict()
    jobs_per_carrier = {k: 0 for k in solution.carriers}
    for ass in (
        sorted_assignments
    ):  # assumes proper sorting in case max_num_bidding_jobs is set
        for bundle, carrier in ass.bundle_to_carrier().items():
            if isinstance(carrier, core_module.carrier.Carrier):
                carrier_id = carrier.id_
            else:
                carrier_id = carrier
            if bundle not in records:
                records[bundle] = dict(bundle=bundle)
            record = records[bundle]
            for check_carrier in solution.carriers:
                if check_carrier in record and record[check_carrier]:
                    continue
                elif check_carrier.id_ == carrier_id:
                    record[check_carrier] = True
                    jobs_per_carrier[check_carrier] += 1
                else:
                    record[check_carrier] = False
        if all([x >= max_num_bidding_jobs for x in jobs_per_carrier.values()]):
            break

    # if not all([x >= max_num_bidding_jobs for x in jobs_per_carrier.values()]):
    #     warnings.warn(f'Not enough ({max_num_bidding_jobs}) bidding jobs were extracted:\n{jobs_per_carrier}')

    bidding_jobs_df = pd.DataFrame.from_records(
        data=list(records.values()), index="bundle"
    )
    assert bidding_jobs_df.index.is_unique
    bidding_jobs_df.sort_index(axis=1, key=lambda x: pd.Index([c.id_ for c in x]))
    return bidding_jobs_df
