import logging
import random
from abc import abstractmethod, ABC
from typing import Sequence, Optional, Union, Callable

import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm, trange

from auction_module.bundle_generation import bundle_generation as bg
from auction_module.bundle_generation.assignment_based.assignment import Assignment
from auction_module.bundle_generation.bundle_based.bundle import Bundle
from auction_module.bundle_generation.fitness import Fitness
from auction_module.bundle_generation.partition_based.partition import Partition
from auction_module.bundle_generation.partition_based.partition_fitness import (
    PartitionFitness,
)
from auction_module.bundling_and_bidding.type_defs import QueriesType
from core_module import instance as it, solution as slt
from core_module.request import Request
from utility_module import combinatorics as cmb
from utility_module.utils import (
    debugger_is_active,
    EXECUTION_START_TIME,
    END_TIME,
    euclidean_distance,
)

logger = logging.getLogger(__name__)

"""
===== NAMING CONVENTION =====

auction_request_pool: Sequence[int][int]
    a sequence of request indices of the requests that were submitted to be auctioned
    
auction_bundle_pool: Sequence[Sequence[int]]
    a nested sequence of request index sequences. Each inner sequence is a {bundle} inside the auction_bundle_pool that
    carriers will have to bid on
    
bundle: Sequence[int]
    a sequence of request indices that make up one bundle -> cannot have duplicates etc., maybe a set rather than a list
    would be better?
    
partition: Sequence[Sequence[int]]
    a sequence of {bundles} (see above) that fully partition the {auction_request_pool}
    
partition_labels: Sequence[int]
    a sequence of bundle indices that partitions the {auction_request_pool}
     NOTE: Contrary to the {partition}, the {partition_labels} is not nested and does not contain request indices but 
     bundle indices
"""


class SinglePartition(bg.BundleGenerationBehavior, ABC):
    pass


class SpatialKMeans(SinglePartition):
    """
    creates a *spatial* k-means partition of the submitted requests.
    generates exactly as many clusters as there are carriers.

    :return partition_labels (not normalized) of the k-means partitioning
    """

    def generate_auction_bundles(
        self,
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        auction_request_pool: tuple[Request],
        num_bidding_jobs: int,
        original_assignment: Assignment,
    ) -> tuple[pd.DataFrame, Optional[Sequence[Partition]]]:
        coords = [(r.x, r.y) for r in auction_request_pool]
        k_means = KMeans(n_clusters=instance.num_carriers).fit(coords)
        partition = Partition(instance, auction_request_pool, k_means.labels_)
        bundle_pool = partition.bundles + original_assignment.bundles()
        bundle_bidding_assignment = pd.DataFrame(
            data=np.full((len(bundle_pool), instance.num_carriers), True),
            index=bundle_pool,
            columns=solution.carriers,
        )
        return bundle_bidding_assignment, None


class SpatialKMeansDepotCentroids(SinglePartition):
    """
    creates a spatial k-means partition of the submitted requests.
    generates exactly as many clusters as there are carriers and uses carrier depots as the initial centroids

    :return partition_labels (nor normalized) of k-means partitioning
    """

    def generate_auction_bundles(
        self,
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        auction_request_pool: tuple[Request],
        num_bidding_jobs: int,
        original_assignment: Assignment,
    ) -> tuple[pd.DataFrame, Optional[Sequence[Partition]]]:
        coords = [(r.x, r.y) for r in auction_request_pool]
        init = np.array([(d.x, d.y) for d in instance.depots])  # depot coordinates
        k_means = KMeans(n_clusters=instance.num_carriers, init=init, n_init=1).fit(
            coords
        )
        partition = Partition(instance, auction_request_pool, k_means.labels_)
        # bundle_pool = list(partition.bundles)
        # bundle_pool.extend(original_partition.bundles)
        bundle_pool = partition.bundles + original_assignment.bundles()
        bundle_bidding_assignment = pd.DataFrame(
            data=np.full((len(bundle_pool), instance.num_carriers), True),
            index=bundle_pool,
            columns=solution.carriers,
        )
        return bundle_bidding_assignment, None


# =====================================================================================================================
# LIMITED NUMBER OF BUNDLES
# =====================================================================================================================
# class LimitedBlockSizePartitions(PartitionBasedBundleGeneration):
#     """
#     Generate a pool of bundles which have a limited bundle size by evaluating different partitions of the auction
#     request pool that respect the maximum bundle size.
#     This might be useful since it can be expected that successfully sold bundles have a medium size and do not
#     contain a very large or small number of requests.
#     """
#     pass


class NumBiddingJobsFromPartitions(bg.BundleGenerationBehavior):
    """
    Generate a pool of bundles that has a limited, predefined number of bundles in it.
    Bundle selection happens by evaluating different partitions of the auction request pool and keeping a partition's
    bundles if the partition's evaluation was sufficiently good.

    Note: the size of the resulting auction pool may differ slightly from the exact given number, since partitions
    are always preserved. Therefore, whenever a partition's evaluation is sufficient, all its bundles will be kept,
    even if this means that the given bundle pool size is slightly exceeded.
    """

    def execute(
        self,
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        auction_request_pool: tuple[Request],
        num_bidding_jobs: int,
        original_assignment: Assignment,
    ) -> tuple[QueriesType, Optional[Sequence[dict[Partition, float]]]]:
        partition_fitness_pool = self.generate_partition_pool(
            instance,
            solution,
            auction_request_pool,
            num_bidding_jobs,
            original_assignment,
        )
        partitions_extracted = self.extract_partitions(partition_fitness_pool)
        bundle_offers = self.extract_bundle_offers(
            instance,
            solution,
            partitions_extracted,
            original_assignment,
            num_bidding_jobs,
        )
        # self.on_execute_end(solution, partitions_extracted)
        return bundle_offers, partitions_extracted

    @abstractmethod
    def generate_partition(
        self,
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        auction_request_pool: tuple[Request],
    ):
        pass

    @abstractmethod
    def generate_partition_pool(
        self,
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        auction_request_pool: tuple[Request],
        num_bidding_jobs: int,
        original_assignment: Assignment,
    ) -> dict[Partition, float]:
        """
        generate a pool of partitions of the auction request pool and evaluate them using the given fitness function.
        :param instance:
        :param solution:
        :param auction_request_pool:
        :param num_bidding_jobs:
        :param original_assignment:
        :return:
        """
        pass

    def extract_partitions(
        self, partition_fitness_pool: dict[Partition, float]
    ) -> dict[Partition, float]:
        return partition_fitness_pool

    def extract_bundle_offers(
        self,
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        partitions_extracted: dict[Partition, float],
        original_assignment: Assignment,
        num_bidding_jobs: int,
    ) -> list[list[tuple[int]]]:
        sorted_partitions = iter(
            sorted(partitions_extracted.items(), key=lambda item: item[1], reverse=True)
        )
        original_partition = Partition.from_assignment(original_assignment).normalize()

        bundles = [b.bitstring for b in original_partition.bundles]
        i = len(bundles)
        while i < num_bidding_jobs:
            part, fitness = next(sorted_partitions)
            for bundle in part.bundles:
                if any(bundle.bitstring) and bundle.bitstring not in bundles:
                    bundles.append(bundle.bitstring)
                    i += 1
        bidding_jobs = [tuple(b) for b in bundles]
        bidding_jobs = [bidding_jobs.copy() for _ in range(instance.num_carriers)]
        return bidding_jobs

    def on_execute_end(
        self,
        solution: slt.CAHDSolution,
        partitions_extracted: dict[Partition, float],
    ):
        # 1 log partition "overlap", i.e. the misclassification error distance (MED) (for Reviewer 2)
        med_dim0 = []
        for partition, fitness in partitions_extracted.items():
            med_dim1 = []
            for other_partition, other_fitness in partitions_extracted.items():
                if partition is other_partition:
                    med = 0
                else:
                    med = partition.misclassification_error(other_partition)
                med_dim1.append(med)
            med_dim0.append(med_dim1)
        mlflow.log_metric("mean_partition_misclassification_error", np.mean(med_dim0))
        pass


# class RandomMaxKPartitions(LimitedNumBundles):
#     """
#     creates random partitions of the submitted bundles, each with AT MOST as many subsets as there are carriers and
#     chooses random ones
#     """
#
#     def generate_auction_bundles(self, instance: it.CAHDInstance, solution: slt.CAHDSolution,
#                                  auction_request_pool: tuple[Request],num_bidding_jobs: int,
#                                  original_assignment: Assignment
#                                  ) -> tuple[pd.DataFrame, Optional[Sequence[Partition]]]:
#
#         # if the number of all possible bundles is lower than then number of required bundles then generate all bundles
#         if 2 ** len(auction_request_pool) - 1 < num_bidding_jobs:
#             bundle_pool = list(set(Bundle(instance, b) for b in cmb.power_set(auction_request_pool, False)))
#             bundle_bidding_assignment = pd.DataFrame(data=np.full((len(bundle_pool), instance.num_carriers), True),
#                                                      index=bundle_pool, columns=solution.carriers)
#             return bundle_bidding_assignment, None
#
#         # otherwise, randomly generate partitions of the data set
#         else:
#             bundle_pool = {b for b in original_assignment.bundles()}
#             partition_pool = {Partition.from_assignment(original_assignment)}
#             for original_bundle in original_assignment.bundles():
#                 bundle_pool.add(original_bundle)
#             with tqdm.tqdm(total=num_bidding_jobs,
#                            disable=not ut.debugger_is_active(),
#                            desc='RandomMaxKPartitions Bundle Generation') as p_bar:
#
#                 while len(bundle_pool) < num_bidding_jobs:
#                     partition = Partition.random(instance, auction_request_pool)
#                     partition.normalize(True)
#                     partition_pool.add(partition)
#                     for bundle in partition.bundles:
#                         if bundle not in bundle_pool:
#                             p_bar.update()
#                             bundle_pool.add(bundle)
#
#             bundle_bid_assignment = pd.DataFrame(data=np.full((len(bundle_pool), instance.num_carriers), True),
#                                                  index=bundle_pool, columns=solution.carriers)
#             return bundle_bid_assignment, tuple(partition_pool)


# class RandomMaxKPartitionsAndSingletons(RandomMaxKPartitions):
#     def generate_auction_bundles(self, instance: it.CAHDInstance, solution: slt.CAHDSolution,
#                                  auction_request_pool: tuple[Request],num_bidding_jobs: int,
#                                  original_assignment: Assignment
#                                  ) -> tuple[pd.DataFrame, Optional[Sequence[Partition]]]:
#         bundle_pool, partition_pool = super().generate_auction_bundles(instance, solution, auction_request_pool,
#                                                                        num_bidding_jobs, original_assignment)
#         bundle_pool = list(bundle_pool)
#
#         for request in auction_request_pool:
#             bundle_pool.append(Bundle(instance, [request]))
#
#         bundle_bidding_assignment = pd.DataFrame(data=np.full((len(bundle_pool), instance.num_carriers), True),
#                                                  index=bundle_pool, columns=solution.carriers)
#         return bundle_bidding_assignment, tuple(partition_pool)


class BestOfManyMaxKPartitions(NumBiddingJobsFromPartitions):
    def __init__(
        self, fitness: Union[PartitionFitness, Fitness], many: int
    ):  # TODO why does this not have a max_k parameter as input?
        super().__init__(fitness)
        self.many = many
        self.name = self.__class__.__name__.replace("Many", str(many))

    @abstractmethod
    def generate_partition(
        self,
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        auction_request_pool,
    ) -> Partition:
        pass

    def generate_partition_pool(
        self,
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        auction_request_pool: tuple[Request],
        num_bidding_jobs: int,
        original_assignment: Assignment,
    ):
        # if the number of all possible bundles is lower than then number of required bundles then generate all bundles
        if 2 ** len(auction_request_pool) - 1 < num_bidding_jobs:
            bundle_pool = list(
                set(
                    Bundle(auction_request_pool, b)
                    for b in cmb.power_set(auction_request_pool, False)
                )
            )
            bundle_bidding_assignment = pd.DataFrame(
                data=np.full((len(bundle_pool), instance.num_carriers), True),
                index=bundle_pool,
                columns=solution.carriers,
            )
            return bundle_bidding_assignment

        # otherwise, generate many partitions of the set
        partitions = dict()
        with tqdm(
            total=self.many,
            disable=not debugger_is_active(),
            desc="Generating partition candidates",
        ) as p_bar:
            while len(partitions) < self.many:
                part = self.generate_partition(instance, solution, auction_request_pool)
                part.normalize(True)
                if part not in partitions:
                    partitions[part] = self.fitness(instance, solution, part)
                    p_bar.update(1)
        return partitions


class BestOfManyRandomMaxKPartitions(BestOfManyMaxKPartitions):
    """
    creates random partitions of the submitted bundles, each with AT MOST as many subsets as there are carriers and
    chooses the best ones according to the given partition valuation
    """

    def generate_partition(
        self,
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        auction_request_pool,
    ) -> Partition:
        return Partition.random_max_k(instance, auction_request_pool)


#
# class BestOfAllMaxKPartitions(LimitedNumBundles):
#     """
#     Creates all possible partitions of the auction request pool of sizes [2, ..., num_carriers] and evaluates them to
#     find the best ones.
#
#     the number of all possible partitions of size k is also known as the Stirling number of the second kind.
#     I can be calculated using sympy's stirling function from sympy.functions.combinatorial.numbers
#
#     """
#
#     def generate_auction_bundles(self, instance: it.CAHDInstance, solution: slt.CAHDSolution,
#                                  auction_request_pool: tuple[Request],num_bidding_jobs: int,
#                                  original_assignment: Assignment
#                                  ) -> tuple[pd.DataFrame, Optional[Sequence[Partition]]]:
#         raise NotImplementedError(f'TODO: size_k_partitions returns partitions rather than partition_labels. '
#                                   f'Labels are requires to create instances of the Partition class')
#         # TODO, return both (1) bundle_pool and (2) partition_pool
#         # TODO adapt to use the new Assignment class rather than the old original_partition
#
#         all_partitions = [Partition(instance, auction_request_pool, [0] * len(auction_request_pool))]
#         for k in range(2, instance.num_carriers + 1):
#             for p in cmb.size_k_partitions(list(auction_request_pool), k):
#                 all_partitions.extend(list(cmb.size_k_partitions(list(auction_request_pool), k)))
#
#         fitness = []
#         for partition in tqdm.tqdm(all_partitions, desc='Bundle Generation', disable=not ut.debugger_is_active()):
#             fitness.append(self.partition_valuation.evaluate_partition(partition, instance, solution))
#
#         sorted_partitions = (partition for _, partition in
#                              sorted(zip(fitness, all_partitions), reverse=True))
#         limited_bundle_pool = [*ut.indices_to_nested_lists(original_partition, auction_request_pool)]
#
#         # TODO loop significantly faster if the limited bundle pool was a set rather than a list! but the return of
#         #  ut.indices_to_nested_list is unhashable: list[list[int]]
#         while len(limited_bundle_pool) < num_bidding_jobs:
#             for bundle in next(sorted_partitions):
#                 if bundle not in limited_bundle_pool:
#                     limited_bundle_pool.append(bundle)
#
#         bundle_bidding_assignment = pd.DataFrame(data=np.ones((len(limited_bundle_pool), instance.num_carriers)),
#                                                  index=limited_bundle_pool, columns=solution.carriers)
#         return bundle_bidding_assignment


class GeneticAlgorithm(NumBiddingJobsFromPartitions):
    """
    Generates max-k-partitions using a Genetic Algorithm to find promising candidates.
    The best partitions of the final population will be put into the bundle pool.
    """

    def __init__(
        self,
        fitness: PartitionFitness,
        num_generations: int,
        population_size: int,
        generation_gap: float,
        mutation_rate: float,
    ):
        super().__init__(fitness)
        self.num_generations = num_generations
        self.population_size = population_size
        self.generation_gap = generation_gap
        self.mutation_rate = mutation_rate
        self.name = (
            self.__class__.__name__
            + f"_ng={num_generations}_ps={population_size}_gg={generation_gap}_mr={mutation_rate}"
        )

    def generate_partition(
        self,
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        auction_request_pool: tuple[Request],
    ):
        pass  # useless for the GA class

    def generate_partition_pool(
        self,
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        auction_request_pool: tuple[Request],
        num_bidding_jobs: int,
        original_assignment: Assignment,
    ) -> dict[Partition, float]:
        # self.fitness.on_before_bundle_gen_start(instance, solution, auction_request_pool)

        # parameters
        # only a fraction of generation_gap is replaced in a new gen. the remaining individuals (generation overlap)
        # are the top (1-generation_gap)*100 % from the previous gen, measured by their fitness
        population: dict[Partition, float] = self.initialize_population(
            instance, solution, auction_request_pool
        )
        for generation_counter in trange(
            self.num_generations,
            desc="Bundle Generation",
            disable=not debugger_is_active(),
        ):
            population = self.generate_new_population(
                instance, solution, population, auction_request_pool
            )
        return population

    def generate_new_population(
        self,
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        previous_population: dict[Partition, float],
        auction_request_pool: tuple[Request],
    ) -> dict[Partition, float]:
        # initialize new generation with the elites from the previous generation
        num_elites = int(self.population_size * (1 - self.generation_gap))
        # select elites with the MAXIMUM fitness
        new_population = sorted(
            previous_population, key=previous_population.get, reverse=True
        )[:num_elites]
        new_population = {k: previous_population[k] for k in new_population}

        offspring_counter = 0
        while offspring_counter < min(
            self.population_size * self.generation_gap,
            2 ** len(auction_request_pool) - 1,
        ):
            # parent selection (get the parent's index first, then the actual parent string/chromosome)
            parent1, parent2 = self._roulette_wheel(previous_population, 2)
            offspring = self.generate_offspring(
                instance, solution, auction_request_pool, parent1, parent2
            )
            if offspring in new_population:
                continue
            else:
                new_population[offspring] = self.fitness(instance, solution, offspring)
                offspring_counter += 1

        return new_population

    # @staticmethod
    # def extract_bundles_from_partitions(
    #         auction_request_pool: tuple[Request],    #         population: dict[Partition, float],
    #         pool_size: int,
    #         original_assignment: Assignment) -> tuple[pd.DataFrame, Optional[Sequence[Partition]]]:
    #     """
    #     create the set of bundles that is offered in the auction based on the final population and its fitness.
    #     Will also include the original bundles.
    #
    #     Note: pool_size may be exceeded to guarantee that ALL bundles of a partition are in the pool (either
    #     all or none can be in the solution).
    #     """
    #     bundle_pool = set(original_assignment.bundles())
    #     partition_pool = {Partition.from_assignment(original_assignment)}
    #     adjusted_pool_size = min(pool_size, 2 ** (len(auction_request_pool)) - 1)
    #     for partition, _ in sorted(population.items(), key=lambda x: x[1], reverse=True):
    #         partition_pool.add(partition)
    #         bundle_pool = bundle_pool.union(partition.bundles)
    #         if len(bundle_pool) >= adjusted_pool_size:
    #             break
    #
    #     return bundle_pool, partition_pool

    def generate_offspring(
        self,
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        auction_request_pool: tuple[Request],
        parent1: Partition,
        parent2: Partition,
    ) -> Partition:
        """
        :param instance:
        :param solution:
        :param auction_request_pool:
        :param parent1:
        :param parent2:

        :return: the NORMALIZED offspring
        """
        # crossover
        crossover_func: Callable = random.choice(
            [
                self._crossover_uniform,
                self._crossover_geo,
                # self._crossover_temporal # TODO uncomment! this must still be tested
            ]
        )
        offspring: Partition = crossover_func(
            instance, solution, auction_request_pool, parent1, parent2
        )
        # normalize
        offspring.normalize(True)

        # mutation
        if random.random() <= self.mutation_rate:
            mutation_func: Callable = random.choice(
                [
                    self._mutation_move,
                    self._mutation_create,
                    self._mutation_join,
                    self._mutation_shift,
                ]
            )
            offspring = mutation_func(
                instance, solution, offspring, auction_request_pool
            )
        offspring.normalize(True)

        return offspring

    def initialize_population(
        self,
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        auction_request_pool: tuple[Request],
    ) -> dict[Partition, float]:
        """
        initializes a population of size population_size. this first generation includes a k-means partition. the
        rest is filled with random partitions of the auction_request_pool

        :param instance:
        :param solution:
        :param auction_request_pool:
        :return: fitness and population
        """

        # initialize one k-means bundle that is likely to be feasible (only location-based)
        coords = [(r.x, r.y) for r in auction_request_pool]
        k_means = KMeans(
            n_clusters=instance.num_carriers,
            n_init="auto",
        ).fit(coords)
        individual = Partition(instance, auction_request_pool, k_means.labels_)
        individual.normalize(True)

        population = {individual: self.fitness(instance, solution, individual)}

        # fill the rest of the population with random max k partitions
        i = 1
        while i < min(self.population_size, 2 ** len(auction_request_pool) - 1):
            individual = Partition.random_max_k(instance, auction_request_pool)
            individual.normalize(True)
            if individual in population:
                continue
            else:
                population[individual] = self.fitness(instance, solution, individual)
                i += 1
        return population

    @staticmethod
    def _roulette_wheel(
        population: dict[Partition, float], n: int = 2
    ) -> dict[Partition, float]:
        weights = np.array(list(population.values()))
        if sum(weights) <= 0:
            # scale fitness values to be positive
            weights = weights + abs(min(weights))
        parents = random.choices(list(population.keys()), weights=weights, k=n)
        return parents

    @staticmethod
    def _crossover_uniform(
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        auction_request_pool: tuple[Request],
        parent1: Partition,
        parent2: Partition,
    ) -> Partition:
        """
        For each request, the corresponding bundle is randomly chosen from parent A or B. This corresponds to the
        uniform crossover of Michalewicz (1996), where only one child is produced.
        """
        offspring_labels = []
        for i in range(parent1.n):
            offspring_labels.append(
                random.choice([parent1.labels[i], parent2.labels[i]])
            )
        return Partition(instance, auction_request_pool, offspring_labels)

    @staticmethod
    def _crossover_temporal(
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        auction_request_pool: tuple[Request],
        parent1: Partition,
        parent2: Partition,
    ) -> Partition:
        """
        combine parents using their time window information
        randomly generate two time points A and B in the execution time horizon. If the opening of a request's TW is
        closer to A it is assigned to the bundle as seen in parent A, and to B otherwise. If the solution consists
        of too many bundles, two randomly chosen ones are merged
        """
        offspring_labels = []
        random_a = random.uniform(EXECUTION_START_TIME, END_TIME)
        random_b = random.uniform(EXECUTION_START_TIME, END_TIME)

        for i, request in enumerate(auction_request_pool):
            delivery_tw_open = instance.tw_open[instance.vertex_from_request(request)]
            dist_a = abs(random_a - delivery_tw_open)
            dist_b = abs(random_b - delivery_tw_open)

            if dist_a < dist_b:
                offspring_labels.append(parent1.labels[i])
            else:
                offspring_labels.append(parent2.labels[i])

        # merge two bundles if there are more bundles than allowed
        if max(offspring_labels) >= instance.num_carriers:
            rnd_bundle_1, rnd_bundle_2 = random.sample(
                range(0, max(offspring_labels) + 1), k=2
            )
            for i in range(len(offspring_labels)):
                if offspring_labels[i] == rnd_bundle_1:
                    offspring_labels[i] = rnd_bundle_2
            assert max(offspring_labels) < instance.num_carriers
        return Partition(instance, auction_request_pool, offspring_labels)

    @staticmethod
    def _crossover_geo(
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        auction_request_pool: tuple[Request],
        parent1: Partition,
        parent2: Partition,
    ) -> Partition:
        """
        "In this operator, we try to keep potentially good parts of existing bundles by combining the parents using
        geographic information. First, we calculate the center of each request, which is the midpoint between pickup
        and delivery location. Then, we randomly generate two points (A and B) in the plane. If the center of a
        request is closer to A, it is assigned to the bundle given in parent A, but if it is closer to B, it gets the
        bundle given in parent B. If the new solution consists of too many bundles, two randomly chosen bundles are
        merged"
        VRP adjusted.
        """
        # setup
        offspring_labels = []
        # two random points in the plane, a and b
        random_a = (
            random.uniform(instance.min_x_coord, instance.max_x_coord),
            random.uniform(instance.min_y_coord, instance.max_y_coord),
        )
        random_b = (
            random.uniform(instance.min_x_coord, instance.max_x_coord),
            random.uniform(instance.min_y_coord, instance.max_y_coord),
        )

        for i, request in enumerate(auction_request_pool):
            dist_a = euclidean_distance((request.x, request.y), *random_a)
            dist_b = euclidean_distance((request.x, request.y), *random_b)

            # copy bundle assignment based on the proximity to the nearest point
            if dist_a < dist_b:
                offspring_labels.append(parent1.labels[i])
            else:
                offspring_labels.append(parent2.labels[i])

        # merge two bundles if there are more bundles than allowed
        if max(offspring_labels) >= instance.num_carriers:
            rnd_bundle_1, rnd_bundle_2 = random.sample(
                range(0, max(offspring_labels) + 1), k=2
            )
            for i in range(len(offspring_labels)):
                if offspring_labels[i] == rnd_bundle_1:
                    offspring_labels[i] = rnd_bundle_2
            assert max(offspring_labels) < instance.num_carriers
        return Partition(instance, auction_request_pool, offspring_labels)

    @staticmethod
    def _mutation_move(
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        offspring: Partition,
        auction_request_pool: tuple[Request],
    ) -> Partition:
        """
        A random number of randomly chosen positions is changed. However, the number of available bundles is not
        increased.
        """
        mutant_labels = list(offspring.labels)
        for pos in random.sample(range(offspring.n), k=random.randint(0, offspring.n)):
            mutant_labels[pos] = random.randint(0, offspring.k - 1)
        return Partition(instance, auction_request_pool, mutant_labels)

    @staticmethod
    def _mutation_create(
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        offspring: Partition,
        auction_request_pool: tuple[Request],
    ) -> Partition:
        """
        A new bundle is created. We randomly chose one request and assign it to the new bundle. If by this the
        maximum number of bundles is exceeded, i.e., if there are more bundles than carriers (see Sect. 4),
        two randomly chosen bundles are merged.
        """
        mutant_labels = list(offspring.labels)
        new_bundle_idx = offspring.k
        # replace random position with new bundle
        mutant_labels[random.randint(0, offspring.n - 1)] = new_bundle_idx

        # merge two random bundles if the new exceeds the num_carriers
        if new_bundle_idx >= instance.num_carriers:
            rnd_bundle_1, rnd_bundle_2 = random.sample(
                range(0, new_bundle_idx + 1), k=2
            )
            for i in range(offspring.n):
                if mutant_labels[i] == rnd_bundle_1:
                    mutant_labels[i] = rnd_bundle_2
        return Partition(instance, auction_request_pool, mutant_labels)

    @staticmethod
    def _mutation_join(
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        offspring: Partition,
        auction_request_pool: tuple[Request],
    ) -> Partition:
        """
        Two randomly chosen bundles are merged. If the offspring has only a single bundle, nothing happens
        """
        mutant_labels = list(offspring.labels)
        num_bundles = offspring.k
        if num_bundles >= 2:
            rnd_bundle_1, rnd_bundle_2 = random.sample(range(0, num_bundles), k=2)
            for i in range(offspring.n):
                if offspring.labels[i] == rnd_bundle_1:
                    mutant_labels[i] = rnd_bundle_2
        return Partition(instance, auction_request_pool, mutant_labels)

    @staticmethod
    def _mutation_shift(
        instance: it.CAHDInstance,
        solution: slt.CAHDSolution,
        offspring: Partition,
        auction_request_pool: tuple[Request],
    ) -> Partition:
        """
        for each of the given bundles in the candidate solution, the centroid is calculated. Then, requests are
        assigned to bundles according to their closeness to the bundle’s centroids.
        """
        mutant_labels = [None] * offspring.n
        centroids = tuple(b.spatial_centroid for b in offspring.bundles)
        for i, request in enumerate(auction_request_pool):
            vertex_coords = (request.x, request.y)

            min_distance = float("inf")
            closest_centroid = None
            for c, centroid in enumerate(centroids):
                distance = euclidean_distance(*vertex_coords, *centroid)
                if distance < min_distance:
                    min_distance = distance
                    closest_centroid = c

            mutant_labels[i] = closest_centroid

        return Partition(instance, auction_request_pool, mutant_labels)
