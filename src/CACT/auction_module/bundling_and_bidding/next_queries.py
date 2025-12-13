import random
from copy import copy, deepcopy
from io import StringIO
from typing import Any, Union, Sequence

import numpy as np
import pandas as pd
from utility_module.parameterized_class import ParameterizedClass
from sklearn.linear_model import Ridge
from tqdm import trange

from auction_module.bundle_generation.assignment_based.assignment import Assignment
from auction_module.bundle_generation.bundle_based.bundle import random_bundle_max_k, Bundle
from auction_module.bundle_generation.partition_based.partition import random_partition, Partition
from auction_module.bundling_and_bidding.fitness_functions.fitness_functions import FitnessFunction
from auction_module.bundling_and_bidding.type_defs import QueriesType
from auction_module.wdp import WDPRidgeRegression
from core_module.carrier import Carrier
from core_module.instance import CAHDInstance
from core_module.request import Request
from utility_module.combinatorics import stirling_second
from utility_module.profiling import track_cumulative_time


class NextQueries(ParameterizedClass):
    def __init__(self, **kwargs):
        self._params = kwargs
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}({self.params})'

    def __str__(self):
        return self.__class__.__name__

    @track_cumulative_time
    def __call__(self, instance: CAHDInstance, carriers: tuple[Carrier], auction_request_pool: tuple[Request],
                 fitness_function: FitnessFunction, num_queries: int, queries: QueriesType) -> QueriesType:
        """

        :param carriers:
        :param auction_request_pool:
        :param instance:
        :param fitness_function:
        :param num_queries: number of queries to return for each bidder
        :param queries: queries that have already been answered. These should be blocklisted from the search
        :return:
        """
        pass


class NextQueriesBundleRandom(NextQueries):
    """
    Generate num_queries random bundles for each bidder.
    """

    @track_cumulative_time
    def __call__(self, instance, carriers, auction_request_pool, fitness_function, num_queries: int,
                 queries: QueriesType) -> QueriesType:
        if not fitness_function.search_space == 'bundle':
            raise RuntimeError(f'bundle fitness function required but {fitness_function.search_space} provided.')
        n = len(queries[0][0])
        next_queries = [[] for _ in range(len(queries))]
        for bidder_idx in range(len(queries)):
            while len(next_queries[bidder_idx]) < num_queries:
                query = Bundle.random(auction_request_pool)
                # do not query the empty bundle
                blocklist = queries[bidder_idx] + next_queries[bidder_idx]
                if query not in blocklist and any(query):
                    next_queries[bidder_idx].append(query)
        return next_queries


class NextQueriesBundleBestOfManyRandomMaxK(NextQueries):
    def __init__(self, many: int, max_k: int = None):
        super().__init__()
        self.many = many
        self.max_k = max_k

    @track_cumulative_time
    def __call__(self, instance, carriers, auction_request_pool, fitness_function, num_queries: int,
                 queries: QueriesType) -> QueriesType:
        if not fitness_function.search_space == 'bundle':
            raise RuntimeError(f'bundle fitness function required but {fitness_function.search_space} provided.')
        if not self.many >= num_queries:
            raise ValueError
        n = len(queries[0][0])
        k = self.max_k if self.max_k is not None else n
        self.max_k = k

        next_queries = [[] for _ in range(len(queries))]
        for bidder_idx in range(len(queries)):
            candidates = dict()
            while len(candidates) < self.many:
                query = random_bundle_max_k(n, k)
                if any(query) and query not in queries[bidder_idx]:
                    candidates[query] = fitness_function(query, bidder_idx=bidder_idx)
            selected_candidates = sorted(candidates.items(), key=lambda item: item[1], reverse=True)[:num_queries]
            next_queries[bidder_idx].extend(x[0] for x in selected_candidates)
        return next_queries


class NextQueriesBundleGeneticAlgorithm(NextQueries):
    def __init__(self,
                 population_size: int = 100,
                 num_generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_probability: float = 0.05,
                 elitism: int = 2,
                 logging: bool = False):
        super().__init__(population_size=population_size, num_generations=num_generations,
                         crossover_rate=crossover_rate,
                         mutation_probability=mutation_probability, elitism=elitism, logging=logging)
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_probability = mutation_probability
        self.elitism = elitism
        self.rng = np.random.default_rng(42)
        self.num_calls = 0
        self.logging = logging
        self.name = f'{self.__class__.__name__}(n={population_size}, g={num_generations}, cr={crossover_rate}, ' \
                    f'mp={mutation_probability}, e={elitism})'

    def initialize_population(self, auction_request_pool: tuple[Request], n: int, blocklist: list[Bundle]):
        """
        initialize population with random bundles
        :param auction_request_pool:
        :param n: number of items
        :param blocklist: bundles that should not be considered
        :return:
        """
        population = dict()
        idx = 0
        while len(population) < self.population_size:
            individual = Bundle.random(auction_request_pool)
            if individual not in blocklist:
                if individual not in population:
                    population[individual] = idx
                    idx += 1
        # return dict keys as list sorted by their index
        return [x[0] for x in sorted(population.items(), key=lambda x: x[1])]

    def select(self, population: list[Assignment], fitness: list[float], num_individuals: int = 2):
        """
        roulette wheel selection: each individual is selected with probability proportional to its fitness
        :param population:
        :param fitness:
        :param num_individuals:
        :return:
        """
        # adjust fitness to be non-negative (required for roulette wheel selection)
        min_fitness = min(fitness)
        epsilon = 1e-6
        adj_fitness = [f + abs(min_fitness) + epsilon for f in fitness]
        adj_fitness_sum = sum(adj_fitness)
        # compute probabilities
        probabilities = [f / adj_fitness_sum for f in adj_fitness]
        return self.rng.choice(population, p=probabilities, size=num_individuals)

    def crossover(self, parent1: Bundle, parent2: Bundle) -> Bundle:
        """
        Uniform crossover: each index's value (binary representation) is inherited from either parent with
        probability 0.5

        :param parent1:
        :param parent2:
        :return:
        """
        k = len(parent1)
        child = [None for _ in range(k)]
        for i in range(k):
            if self.rng.random() < 0.5:
                child[i] = parent1.bitstring[i]
            else:
                child[i] = parent2.bitstring[i]
        return Bundle.from_binary(parent1.all_items, child)

    def mutate(self, individual: Bundle):
        """
        Bit Flip mutation: each number is changed with probability
        :param individual:
        :return:
        """
        k = len(individual)
        mutant = list(individual.bitstring)
        for i in range(k):
            if self.rng.random() < self.mutation_probability:
                mutant[i] = 1 - mutant[i]
        return Bundle.from_binary(individual.all_items, mutant)

    def evolve_population(self, current_population: list[Assignment], current_fitness: list[float],
                          blocklist: list[Bundle]):
        new_population = dict()
        idx = 0
        # elitism: keep best individuals
        best_individuals = sorted(zip(current_population, current_fitness), key=lambda x: x[1], reverse=True)[
                           :self.elitism]
        for individual, fitness in best_individuals:
            new_population[individual] = idx
            idx += 1

        # crossover and mutation to generate new individuals
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select(current_population, current_fitness, num_individuals=2)
            if self.rng.random() < self.crossover_rate:
                child = self.crossover(parent1, parent2)
            else:
                child = copy(parent1)

            child = self.mutate(child)
            if any(child.requests):  # and child not in current_population:
                if child not in blocklist:
                    if child not in new_population:
                        new_population[child] = idx
                        idx += 1
        return [x[0] for x in sorted(new_population.items(), key=lambda x: x[1])]

    @track_cumulative_time
    def __call__(self, instance: CAHDInstance, carriers: tuple[Carrier], auction_request_pool: tuple[Request],
                 fitness_function: FitnessFunction, num_queries: int, queries: QueriesType) -> QueriesType:
        if not fitness_function.search_space == 'bundle':
            raise RuntimeError(f'bundle fitness function required but {fitness_function.search_space} provided.')
        if self.logging:
            self.run['fitness_function'] = fitness_function.name
        n = len(queries[0][0])

        next_queries = [[] for _ in range(len(queries))]
        for bidder_idx in range(len(queries)):
            blocklist = queries[bidder_idx]
            current_population = self.initialize_population(auction_request_pool, n, blocklist)
            for generation_counter in range(self.num_generations):
                current_fitness = fitness_function(instance, current_population, bidder_idx=bidder_idx)
                if self.logging:
                    self.run[f'bidder_{bidder_idx}/avg_fitness'].append(np.mean(current_fitness))
                    self.run[f'bidder_{bidder_idx}/max_fitness'].append(max(current_fitness))
                    self.run[f'bidder_{bidder_idx}/min_fitness'].append(min(current_fitness))
                    df = pd.DataFrame(data={'fitness': current_fitness, 'bundle': current_population})
                    csv_buffer = StringIO()
                    df.to_csv(csv_buffer, index=True)
                    self.run[f'call_{self.num_calls}/bidder_{bidder_idx}/fitness_gen_{generation_counter}'].upload(
                        File.from_stream(csv_buffer, extension='csv'))
                current_population = self.evolve_population(current_population, current_fitness, blocklist)

            sorted_candidates = sorted(zip(current_population, current_fitness), key=lambda x: x[1], reverse=True)
            while len(next_queries[bidder_idx]) < num_queries:
                individual, fitness = sorted_candidates.pop(0)
                if any(individual.requests) and individual not in queries[bidder_idx]:
                    next_queries[bidder_idx].append(individual)
        if self.logging:
            self.run.wait()
        self.num_calls += 1
        return next_queries


# ======================================================================================================================
def binary_to_rgs(binary: Union[Partition, Assignment]) -> tuple[int]:
    """
    convert binary representation of a Partition or Assignment to restricted growth string representation
    :param binary:
    :return:
    """
    n = len(list(binary)[0])
    rgs = [None for _ in range(n)]
    for i, bundle in enumerate(sorted(binary, reverse=True)):
        for j, item in enumerate(bundle):
            if item:
                rgs[j] = i
    return tuple(rgs)


def rgs_to_binary(rgs: Sequence[int], k: int) -> Assignment:
    """
    convert restricted growth string representation of a partition to binary representation
    :param rgs:
    :param k:
    :return:
    """
    n = len(rgs)
    binary_partition = [[0 for _ in range(n)] for _ in range(k)]
    for i, r in enumerate(rgs):
        binary_partition[r][i] = 1
    return tuple(tuple(bundle) for bundle in binary_partition)


class NextQueriesPartition(NextQueries):
    @staticmethod
    def extract_next_queries(candidates: list[tuple[Any, float]], num_queries: int, queries: QueriesType):
        # Extract final queries for each bidder
        next_queries = [[] for _ in range(len(queries))]
        candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)
        while any(len(x) < num_queries for x in next_queries):
            partition, fitness = candidates_sorted.pop(0)
            # assert isinstance(partition, Partition)
            for bidder_idx in range(len(queries)):
                for query in partition:
                    blocklist = queries[bidder_idx] + next_queries[bidder_idx]
                    # if len(next_queries[bidder_idx]) < num_queries and query not in blocklist and any(query):
                    if query not in blocklist and any(query):
                        next_queries[bidder_idx].append(query)
        return next_queries


class NextQueriesPartitionRandom(NextQueriesPartition):
    """
    every bidder will get the same queries. incomplete partitions may be queried to ensure that exactly num_queries
    queries are returned for each bidder.
    """

    @track_cumulative_time
    def __call__(self, instance, carriers, auction_request_pool, fitness_function, num_queries: int,
                 queries: QueriesType) -> QueriesType:
        if not fitness_function.search_space == 'partition':
            raise RuntimeError(f'partition fitness function required but {fitness_function.search_space} provided.')
        n = len(queries[0][0])
        next_queries = [[] for _ in range(len(queries))]
        while any(len(x) < num_queries for x in next_queries):
            partition = random_partition(n, len(queries))
            for query in partition:
                if any(query):  # do not query the empty bundle
                    for bidder_idx in range(len(queries)):
                        blocklist = queries[bidder_idx] + next_queries[bidder_idx]
                        if len(next_queries[bidder_idx]) < num_queries and query not in blocklist:
                            next_queries[bidder_idx].append(query)
        return next_queries


class NextQueriesPartitionBestOfManyRandom(NextQueriesPartition):
    def __init__(self, many: int):
        super().__init__()
        self.many = many

    @track_cumulative_time
    def __call__(self, instance, carriers, auction_request_pool, fitness_function, num_queries: int,
                 queries: QueriesType) -> QueriesType:
        if not fitness_function.search_space == 'partition':
            raise RuntimeError(f'partition fitness function required but {fitness_function.search_space} provided.')
        assert self.many >= num_queries
        n = len(queries[0][0])

        candidates = dict()
        # Generate candidate queries with random partitions
        while len(candidates) < self.many:
            partition: Partition = random_partition(n, len(queries))
            candidates[tuple(sorted(partition))] = fitness_function(partition)

        next_queries = self.extract_next_queries(list(candidates.items()), num_queries, queries)

        assert all(len(x) == num_queries for x in next_queries)
        return next_queries


class NextQueriesPartitionGeneticAlgorithm(NextQueriesPartition):
    def __init__(self,
                 population_size: int = 100,
                 num_generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_probability: float = 0.05,
                 elitism: int = 2,
                 logging: bool = False):
        super().__init__(population_size=population_size, num_generations=num_generations,
                         crossover_rate=crossover_rate, mutation_probability=mutation_probability, elitism=elitism,
                         logging=logging)
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_probability = mutation_probability
        self.elitism = elitism
        self.logging = logging

    def __repr__(self):
        return (f'{self.__class__.__name__}(n={self.population_size}, g={self.num_generations},'
                f' cr={self.crossover_rate}, mp={self.mutation_probability}, e={self.elitism})')

    def initialize_population(self, n: int, k: int):
        population = set()
        while len(population) < self.population_size:
            population.add(tuple(random_partition(n, k)))
        return list(population)

    def select(self, population: list[Partition], fitness: list[float], num_individuals: int = 2):
        """
        roulette wheel selection: each individual is selected with probability proportional to its fitness.
        :param population:
        :param fitness:
        :param num_individuals:
        :return:
        """
        # adjust fitness to be non-negative
        epsilon = 1e-6
        min_fitness = min(fitness)
        if min_fitness == -np.inf:  # edge case
            # exclude individuals with -inf fitness
            population_new, fitness_new = zip(*[(p, f) for p, f in zip(population, fitness) if f != -np.inf])
            if len(population_new) < num_individuals:
                # select k random individuals out of those that have -inf fitness
                population_inf, fitness_inf = zip(*[(p, f) for p, f in zip(population, fitness) if f == -np.inf])
                population_add = random.sample(population_inf, num_individuals - len(population_new))
                population_new += tuple(population_add)
                fitness_new += tuple([min(fitness_new) - epsilon] * len(population_add))
            return self.select(population_new, fitness_new, num_individuals)
        adj_fitness = [f + abs(min_fitness) + epsilon for f in fitness]
        adj_fitness_sum = sum(adj_fitness)

        # edge cases with infinite or nan fitness
        if adj_fitness_sum == np.inf:
            # select k random individuals out of those that have infinite fitness
            fitness_new = [1 if f == np.inf else 0 for f in fitness]
            return self.select(population, fitness_new, num_individuals)
        if np.isnan(adj_fitness_sum):
            if all(np.isnan(f) for f in fitness):
                # select k random individuals out of those that have nan fitness
                fitness_new = [1] * len(fitness)
                return self.select(population, fitness_new, num_individuals)
            else:
                # exclude individuals with nan fitness
                population_new, fitness_new = zip(*[(p, f) for p, f in zip(population, fitness) if not np.isnan(f)])
                if len(population_new) < num_individuals:
                    # select k - len(population_new) random individuals out of those that have nan fitness
                    population_nan, fitness_nan = zip(*[(p, f) for p, f in zip(population, fitness) if np.isnan(f)])
                    population_add = random.sample(population_nan, num_individuals - len(population_new))
                    population_new += tuple(population_add)
                    fitness_new += tuple([min(fitness_new) - epsilon] * len(population_add))
                return self.select(population_new, fitness_new, num_individuals)

        # compute probabilities
        probabilities = [f / adj_fitness_sum for f in adj_fitness]
        return random.choices(population, weights=probabilities, k=num_individuals)

    def crossover(self, parent1: Partition, parent2: Partition):
        """
        Uniform crossover: each number (RGS representation) is inherited from either parent with probability 0.5
        :param parent1:
        :param parent2:
        :return:
        """
        parent1_rgs = binary_to_rgs(parent1)
        parent2_rgs = binary_to_rgs(parent2)
        n = len(parent1_rgs)
        k = len(parent1)
        child_rgs = [None for _ in range(n)]
        for i in range(n):
            if np.random.rand() < 0.5:
                child_rgs[i] = parent1_rgs[i]
            else:
                child_rgs[i] = parent2_rgs[i]
        child = rgs_to_binary(tuple(child_rgs), k)
        return child

    def mutate(self, individual: Partition):
        """
        Flip mutation: each number (RGS representation) is changed with probability mutation_probability
        :param individual:
        :return:
        """
        individual_rgs = list(binary_to_rgs(individual))
        n = len(individual_rgs)
        k = len(individual)
        for i in range(n):
            if np.random.rand() < self.mutation_probability:
                individual_rgs[i] = random.sample(set(range(k)).difference({individual_rgs[i]}), 1)[0]
        individual = rgs_to_binary(tuple(individual_rgs), k)
        return individual

    def evolve_population(self, current_population: list[Partition], current_fitness: list[float]):
        new_population = set()
        # elitism: keep best individuals
        best_individuals = sorted(zip(current_population, current_fitness), key=lambda x: x[1], reverse=True)[
                           :self.elitism]
        new_population = new_population.union([x[0] for x in best_individuals])

        # crossover and mutation to generate new individuals
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select(current_population, current_fitness, num_individuals=2)
            if np.random.rand() < self.crossover_rate:
                child = self.crossover(parent1, parent2)
            else:
                child = copy(parent1)

            child = self.mutate(child)
            new_population.add(child)
        return list(new_population)

    @track_cumulative_time
    def __call__(self, instance, carriers, auction_request_pool, fitness_function, num_queries: int,
                 queries: QueriesType) -> QueriesType:
        if not fitness_function.search_space == 'partition':
            raise RuntimeError(f'partition fitness function required but {fitness_function.search_space} provided.')
        if self.logging:
            self.run['fitness_function'] = fitness_function.name
        n = len(queries[0][0])
        k = len(queries)

        current_population = self.initialize_population(n, k)
        for generation_counter in range(self.num_generations):
            current_fitness = fitness_function(current_population)
            if self.logging:
                self.run[f'avg_fitness'].append(np.mean(current_fitness))
                self.run[f'max_fitness'].append(max(current_fitness))
                self.run[f'min_fitness'].append(min(current_fitness))
            current_population = self.evolve_population(current_population, current_fitness)

        population_and_fitness = list(zip(current_population, fitness_function(current_population)))
        next_queries = self.extract_next_queries(population_and_fitness, num_queries, queries)
        if self.logging:
            self.run.wait()
        return next_queries


# ======================================================================================================================
class NextQueriesAssignment(NextQueries):
    @staticmethod
    def extract_next_queries(candidates: list[tuple[Assignment, float]], num_queries: int,
                             queries: QueriesType) -> QueriesType:
        # Generate final queries for each bidder
        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        next_queries = [[] for _ in range(len(queries))]
        while any(len(x) < num_queries for x in next_queries):
            try:
                assignment, _ = sorted_candidates.pop(0)
            except IndexError:
                # not enough candidates left, generate random assignment
                assignment = Assignment.random(candidates[0][0]._carriers, candidates[0][0].requests())

            for carrier, bundle in assignment.carrier_to_bundle().items():
                bidder_idx = carrier.id_
                if len(next_queries[bidder_idx]) < num_queries:
                    # TODO increase incrementally instead of re-creating every iteration:
                    blocklist = queries[bidder_idx] + next_queries[bidder_idx]
                    if any(bundle.requests) and bundle not in blocklist:
                        next_queries[bidder_idx].append(bundle)
        return next_queries


class NextQueriesAssignmentRandom(NextQueriesPartition):

    @track_cumulative_time
    def __call__(self,
                 instance: CAHDInstance,
                 carriers: tuple[Carrier],
                 auction_request_pool: tuple[Request],
                 fitness_function: FitnessFunction,
                 num_queries: int,
                 queries: QueriesType) -> QueriesType:
        if not fitness_function.search_space == 'assignment':
            raise RuntimeError(f'assignment fitness function required but {fitness_function.search_space} provided.')
        next_queries = [[] for _ in range(len(queries))]
        while any(len(x) < num_queries for x in next_queries):
            assignment = Assignment.random(carriers, auction_request_pool)
            for carrier, bundle in assignment.carrier_to_bundle().items():
                if any(bundle.requests):  # do not query the empty bundle
                    # TODO queries should be dict[Carrier, tuple[Bundle]] (responses equivalently)
                    blocklist = queries[carrier.id_] + next_queries[carrier.id_]
                    if len(next_queries[carrier.id_]) < num_queries and bundle not in blocklist:
                        next_queries[carrier.id_].append(bundle)
        return next_queries


class NextQueriesAssignmentBestOfManyRandom(NextQueriesAssignment):
    def __init__(self, many: int):
        super().__init__()
        self.many = many

    @track_cumulative_time
    def __call__(self, instance: CAHDInstance, carriers: tuple[Carrier], auction_request_pool: tuple[Request],
                 fitness_function: FitnessFunction, num_queries: int, queries: QueriesType) -> QueriesType:
        if not fitness_function.search_space == 'assignment':

            raise RuntimeError(f'assignment fitness function required but {fitness_function.search_space} provided.')

        if not self.many >= num_queries:
            raise ValueError()
        n = len(queries[0][0])

        candidates = dict()
        # Generate many candidate queries with random assignments
        while len(candidates) < self.many:
            assignment: Assignment = Assignment.random(carriers, auction_request_pool)
            candidates[tuple(sorted(assignment))] = fitness_function(instance, [assignment])

        next_queries = self.extract_next_queries(list(candidates.items()), num_queries, queries)

        assert all(len(x) == num_queries for x in next_queries)
        return next_queries


class NextQueriesAssignmentGeneticAlgorithm(NextQueriesAssignment):
    def __init__(self,
                 population_size: int = 100,
                 num_generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_probability: float = 0.05,
                 elitism: int = 2,
                 logging: bool = False):
        super().__init__(population_size=population_size, num_generations=num_generations,
                         crossover_rate=crossover_rate,
                         mutation_probability=mutation_probability, elitism=elitism, logging=logging)
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_probability = mutation_probability
        self.elitism = elitism
        self.logging = logging

    def __repr__(self):
        return (f'{self.__class__.__name__}(n={self.population_size}, g={self.num_generations},'
                f' cr={self.crossover_rate}, mp={self.mutation_probability}, e={self.elitism})')

    def initialize_population(self, carriers: tuple[Carrier], auction_request_pool: tuple[Request]) -> list[Assignment]:
        """
        generate a list of random assignments in binary form!
        :param carriers:
        :param auction_request_pool:
        :return:
        """
        population = []
        while len(population) < self.population_size:
            assignment = Assignment.random(carriers, auction_request_pool)
            if assignment not in population:  # can't use set (for faster check) because assignment is not hashable
                population.append(assignment)
        return population

    def select(self, population: list[Assignment], fitness: list[float], num_individuals: int = 2) -> list[Assignment]:
        """
        roulette wheel selection: each individual is selected with probability proportional to its fitness.
        :param population:
        :param fitness:
        :param num_individuals:
        :return:
        """
        # adjust fitness to be non-negative
        epsilon = 1e-6
        min_fitness = min(fitness)
        if min_fitness == -np.inf:  # edge case
            # exclude individuals with -inf fitness
            population_new, fitness_new = zip(*[(p, f) for p, f in zip(population, fitness) if f != -np.inf])
            if len(population_new) < num_individuals:
                # select k random individuals out of those that have -inf fitness
                population_inf, fitness_inf = zip(*[(p, f) for p, f in zip(population, fitness) if f == -np.inf])
                population_add = random.sample(population_inf, num_individuals - len(population_new))
                population_new += tuple(population_add)
                fitness_new += tuple([min(fitness_new) - epsilon] * len(population_add))
            return self.select(population_new, fitness_new, num_individuals)
        adj_fitness = [f + abs(min_fitness) + epsilon for f in fitness]
        adj_fitness_sum = sum(adj_fitness)

        # edge cases with infinite or nan fitness
        if adj_fitness_sum == np.inf:
            # select k random individuals out of those that have infinite fitness
            fitness_new = [1 if f == np.inf else 0 for f in fitness]
            return self.select(population, fitness_new, num_individuals)
        if np.isnan(adj_fitness_sum):
            if all(np.isnan(f) for f in fitness):
                # select k random individuals out of those that have nan fitness
                fitness_new = [1] * len(fitness)
                return self.select(population, fitness_new, num_individuals)
            else:
                # exclude individuals with nan fitness
                population_new, fitness_new = zip(*[(p, f) for p, f in zip(population, fitness) if not np.isnan(f)])
                if len(population_new) < num_individuals:
                    # select k - len(population_new) random individuals out of those that have nan fitness
                    population_nan, fitness_nan = zip(*[(p, f) for p, f in zip(population, fitness) if np.isnan(f)])
                    population_add = random.sample(population_nan, num_individuals - len(population_new))
                    population_new += tuple(population_add)
                    fitness_new += tuple([min(fitness_new) - epsilon] * len(population_add))
                return self.select(population_new, fitness_new, num_individuals)

        # compute probabilities
        probabilities = [f / adj_fitness_sum for f in adj_fitness]
        return random.choices(population, weights=probabilities, k=num_individuals)

    def crossover(self, parent1: Assignment, parent2: Assignment) -> Assignment:
        """
        Uniform crossover: each number (RGS representation) is inherited from either parent with probability 0.5
        :param parent1:
        :param parent2:
        :return:
        """
        parent1_rgs = parent1.as_rgs()
        parent2_rgs = parent2.as_rgs()
        n = len(parent1_rgs)
        # k = len(parent1._carriers)
        child_rgs = [None for _ in range(n)]
        for i in range(n):
            if np.random.rand() < 0.5:
                child_rgs[i] = parent1_rgs[i]
            else:
                child_rgs[i] = parent2_rgs[i]
        child = Assignment.from_rgs(parent1._carriers, parent1.requests(), child_rgs)
        return child

    def mutate(self, individual: Assignment):
        """
        Flip mutation: each number (RGS representation) is changed with probability mutation_probability
        :param individual:
        :return:
        """
        rgs = list(individual.as_rgs())
        num_requests = len(rgs)
        k = len(individual._carriers)
        for request_idx in range(num_requests):
            if np.random.rand() < self.mutation_probability:
                currently_assigned_carrier_idx = rgs[request_idx]
                sampling_set = set(range(k)).difference({currently_assigned_carrier_idx})
                rgs[request_idx] = random.sample(sampling_set, 1)[0]
        mutant = Assignment.from_rgs(individual._carriers, individual.requests(), rgs)
        return mutant

    def evolve_population(self, current_population: list[Assignment], current_fitness: list[float]) -> list[Assignment]:
        new_population = []
        # elitism: keep the best individuals, i.e. those with highest fitness
        sorted_individuals = sorted(zip(current_population, current_fitness), key=lambda x: x[1], reverse=True)
        elite_individuals = sorted_individuals[:self.elitism]
        new_population += [x[0] for x in elite_individuals]

        # crossover and mutation to generate new individuals
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select(current_population, current_fitness, num_individuals=2)
            if np.random.rand() < self.crossover_rate:
                child = self.crossover(parent1, parent2)
            else:
                child = copy(parent1)
            child = self.mutate(child)
            new_population.append(child)
        return list(new_population)

    @track_cumulative_time
    def __call__(self, instance: CAHDInstance, carriers: tuple[Carrier], auction_request_pool: tuple[Request],
                 fitness_function: FitnessFunction, num_queries: int, queries: QueriesType) -> QueriesType:
        if not fitness_function.search_space == 'assignment':
            raise RuntimeError(f'assignment fitness function required but {fitness_function.search_space} provided.')

        if self.logging:
            self.run['fitness_function'] = fitness_function.__class__.__name__
        n = len(queries[0][0].all_items)  # number of requests in the auction pool
        k = len(carriers)  # number of carriers to assign to
        if not self.population_size <= sum(stirling_second(n, k_ + 1) for k_ in range(k)):
            raise ValueError
        current_population = self.initialize_population(carriers, auction_request_pool)
        for _ in trange(self.num_generations, disable=True):
            # current_fitness = [fitness_function(x) for x in current_population]
            current_fitness = fitness_function(instance, current_population)
            current_population = self.evolve_population(current_population, current_fitness)
            if self.logging:
                self.run[f'avg_fitness'].append(np.mean(current_fitness))
                self.run[f'max_fitness'].append(max(current_fitness))
                self.run[f'min_fitness'].append(min(current_fitness))

        population_and_fitness = list(zip(current_population, current_fitness))
        next_queries = self.extract_next_queries(population_and_fitness, num_queries, queries)
        if self.logging:
            self.run.wait()
        return next_queries


# ======================================================================================================================

class NextQueriesAssignmentBLS2021(NextQueriesAssignment):
    """
    Based on Brero, Lubin et al. 2021 – Machine Learning-powered Iterative Combinatorial Auctions.
    Select the next queries by solving the WDP using the estimated valuation models. In the paper, they introduce
    WDP formulations that allow to solve the problem efficiently using linear regression and SVR estimators.
    """

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    @track_cumulative_time
    def __call__(self, instance, carriers, auction_request_pool, fitness_function, num_queries: int,
                 queries: QueriesType) -> QueriesType:
        if not fitness_function.search_space == 'assignment':
            raise RuntimeError(f'assignment fitness function required but {fitness_function.search_space} provided.')

        # assuming that the fitness function uses ridge regression to estimate the valuation functions
        assert isinstance(fitness_function.models[0], Ridge), 'Fitness function must use Ridge'
        assert (fitness_function.models[0].fit_intercept is False)
        blocklist = deepcopy(queries)
        num_carriers = len(queries)
        next_queries = [[] for _ in range(num_carriers)]
        carrier_needs_queries = [True for _ in range(num_carriers)]
        while any(carrier_needs_queries):
            wdp = WDPRidgeRegression(fitness_function.models, 'min', blocklist, carrier_needs_queries, 1)
            for carrier_idx in range(num_carriers):
                query = wdp.item_allocation[carrier_idx]
                if any(query):  # do not query the empty bundle
                    next_queries[carrier_idx].append(query)
                    blocklist[carrier_idx].append(query)
                    carrier_needs_queries[carrier_idx] = len(next_queries[carrier_idx]) < num_queries

        return next_queries
