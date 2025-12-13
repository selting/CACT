from abc import abstractmethod
from typing import Sequence, Optional, Union

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB
from tqdm import tqdm

import utility_module.combinatorics as cmb
from auction_module.bundle_generation import bundle_generation as bg
from auction_module.bundle_generation.assignment_based.assignment import Assignment
from auction_module.bundle_generation.assignment_based.assignment_based_bg import assignments_to_bidding_job_df
from auction_module.bundle_generation.bundle_based.bundle import Bundle
from auction_module.bundle_generation.bundle_based.bundle_fitness import BundleFitness
from auction_module.bundle_generation.fitness import Fitness
from auction_module.bundle_generation.partition_based.partition import Partition
from auction_module.bundling_and_bidding.type_defs import QueriesType
from core_module import instance as it, solution as slt
from core_module.request import Request
from utility_module.utils import debugger_is_active


class AllBundles(bg.BundleGenerationBehavior):
    """
    creates the power set of all the submitted requests, i.e. all subsets of size k for all k = 1, ..., len(pool).
    Does not include emtpy set.
    """

    def __init__(self):
        super().__init__(None)

    def execute(self,
                instance: it.CAHDInstance,
                solution: slt.CAHDSolution,
                auction_request_pool: tuple[Request],                num_bidding_jobs,
                original_assignment: Assignment) -> pd.DataFrame:
        all_bundles = tuple(Bundle(auction_request_pool, b) for b in tqdm(cmb.power_set(auction_request_pool, False),
                                                      total=2 ** (len(auction_request_pool))))
        bundle_bidding_assignment = pd.DataFrame(data=np.ones(len(all_bundles), instance.num_carriers),
                                                 index=all_bundles, columns=solution.carriers)
        return bundle_bidding_assignment, None


class BestOfAllBundles(bg.BundleGenerationBehavior):
    """
    Generates the power set of all the submitted requests and selects those that have the best valuation according
    to the bundle valuation function
    """

    def generate_auction_bundles(self,
                                 instance: it.CAHDInstance,
                                 solution: slt.CAHDSolution,
                                 auction_request_pool: tuple[Request],                                 num_bidding_jobs,
                                 original_assignment: Assignment) -> pd.DataFrame:
        all_bundles = tuple(Bundle(auction_request_pool, b) for b in (cmb.power_set(auction_request_pool, False)))
        best_bundles = sorted(all_bundles, key=self.bundle_valuation)[:num_bidding_jobs]
        bundle_bidding_assignment = pd.DataFrame(data=np.ones(len(best_bundles), instance.num_carriers),
                                                 index=best_bundles, columns=solution.carriers)
        return bundle_bidding_assignment, None


class BestOfManyMaxNBundles(bg.BundleGenerationBehavior):
    """
    Generate many bundles with at most n requests and select the best ones according to the bundle fitness
    """

    def __init__(self, fitness: Union[BundleFitness, Fitness], many: int, max_n: Optional[int]):
        """

        :param fitness:
        :param many:
        :param max_n: If None, then the maximum number of requests per bundle is limited by the number of requests in
        the auction request pool
        """
        super().__init__(fitness)
        self.many = many
        self.max_n = max_n
        name = self.__class__.__name__.replace('Many', str(many))
        if self.max_n is not None:
            name = name.replace('MaxN', f'Max{max_n}')
        self.name = name

    def execute(self,
                instance: it.CAHDInstance,
                solution: slt.CAHDSolution,
                auction_request_pool: tuple[Request],                num_bidding_jobs: int,
                original_assignment: Assignment
                ) -> tuple[QueriesType, Optional[Sequence[dict[Partition, float]]]]:
        # if the number of all possible bundles is lower than then number of required bundles then generate all bundles
        if 2 ** len(auction_request_pool) - 1 < num_bidding_jobs:
            raise ValueError(f'Number of bidding jobs ({num_bidding_jobs}) exceeds the number of possible bundles '
                             f'({2 ** len(auction_request_pool) - 1})')

        if self.max_n is None:
            self.max_n = len(auction_request_pool)

        bundles = dict()
        for b in original_assignment.bundles():
            fitness = float('inf')
            bundles[b.bitstring] = [fitness for _ in range(instance.num_carriers)]

        with tqdm(total=self.many, desc=f'Generating bundle candidates', disable=not debugger_is_active()) as pbar:
            while len(bundles) < self.many:
                bundle = self.generate_bundle(instance, solution, auction_request_pool)
                if not any(bundle):
                    # NOTE skip empty bundle. important because computing the bid on the empty bundle is different to
                    #  the post-request-selection solution and can cause the post-auction solution to be worse than
                    #  the pre-auction solution
                    continue
                if bundle not in bundles:
                    fitness = self.fitness(instance, solution, bundle)
                    if isinstance(fitness, float):
                        bundles[bundle] = [fitness for _ in range(instance.num_carriers)]
                    else:
                        bundles[bundle] = fitness
                    pbar.update()

        bundle_offers = self.extract_bundle_offers(instance, solution, original_assignment, bundles, num_bidding_jobs)
        return bundle_offers, None

    @abstractmethod
    def generate_bundle(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, auction_request_pool:tuple[Bundle]):
        pass

    def extract_bundle_offers(self, instance, solution, original_assignment, bundles: dict[Bundle, float],
                              num_bidding_jobs) -> QueriesType:
        bundle_offers = []
        for carrier_idx in range(instance.num_carriers):
            combined = sorted(bundles.items(), key=lambda x: x[1][carrier_idx], reverse=True)
            bundles_sorted, fitness_sorted = zip(*combined)
            bundles_sorted = list(bundles_sorted)
            carrier_offers = bundles_sorted[:num_bidding_jobs].copy()
            # TODO add the original bundle AFTER taking num_bidding_jobs bundles from the sorted list?
            bundle_offers.append(carrier_offers)
        return bundle_offers


class BestOfManyRandomMaxNBundles(BestOfManyMaxNBundles):
    def __init__(self, fitness: Union[BundleFitness, Fitness], many: int, max_n: Optional[int]):
        super().__init__(fitness, many, max_n)
        from utility_module.random import rng
        self.rng = rng

    def generate_bundle(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, auction_request_pool: tuple[Request]):
        return Bundle.random(auction_request_pool)


class BestOfManyRandomMaxNBundlesAndSetPartitioning(BestOfManyRandomMaxNBundles):
    """
    Instead of extracting bundles from the searched set based on their individual fitness, we solve the exact cover/
    set partitioning problem and extract bundles from the best found solutions.
    """

    def get_bidding_jobs_df(self, instance, solution, original_assignment, bundles, num_bidding_jobs):
        return bundles_to_bidding_job_df_set_partitioning(instance, solution, original_assignment, bundles,
                                                          num_bidding_jobs)


class BestOfManyRandomMaxNBundlesAndWDP(BestOfManyRandomMaxNBundles):
    """
    Instead of extracting bundles from the searched set based on their individual fitness, we solve the winner
    determination problem and extract bundles from the best found solutions.
    """

    def get_bidding_jobs_df(self, instance, solution, original_assignment, bundles, num_bidding_jobs):
        return bundles_to_bidding_job_df_wdp(instance, solution, original_assignment, bundles,
                                             num_bidding_jobs)


def bundles_to_bidding_job(instance: it.CAHDInstance,
                           solution: slt.CAHDSolution,
                           original_assignment: Assignment,
                           bundles: dict[tuple[int], float],
                           num_bidding_jobs: int) -> list[list[tuple[int]]]:
    """
    Selects the best bundles from a dictionary of bundles and their fitness values
    :param instance:
    :param solution:
    :param original_assignment:
    :param bundles:
    :param num_bidding_jobs:
    :return:
    """
    combined = sorted(bundles.items(), key=lambda x: x[1], reverse=True)
    bundles_sorted, fitness_sorted = zip(*combined)
    bundles_sorted = original_assignment.bundles() + bundles_sorted
    bundle_bidding_jobs = [bundles_sorted[:num_bidding_jobs].copy for _ in range(instance.num_carriers)]
    return bundle_bidding_jobs


def bundles_to_bidding_job_df(instance: it.CAHDInstance,
                              solution: slt.CAHDSolution,
                              original_assignment: Assignment,
                              bundles: dict,
                              num_bidding_jobs: int) -> pd.DataFrame:
    """
    Selects the best bundles from a dictionary of bundles and their fitness values and returns a bidding job
    dataframe in which every carrier has to bid on every bundle
    :param instance:
    :param solution:
    :param original_assignment:
    :param bundles:
    :param num_bidding_jobs:
    :return:
    """
    combined = sorted(bundles.items(), key=lambda x: x[1], reverse=True)
    bundles_sorted, fitness_sorted = zip(*combined)
    bundles_sorted = original_assignment.bundles() + bundles_sorted
    bundle_bidding_jobs = pd.DataFrame(
        data=np.full((num_bidding_jobs, instance.num_carriers), True),
        index=bundles_sorted[:num_bidding_jobs],
        columns=solution.carriers)
    assert bundle_bidding_jobs.index.is_unique
    return bundle_bidding_jobs


def bundles_to_bidding_job_df_set_partitioning(instance: it.CAHDInstance,
                                               solution: slt.CAHDSolution,
                                               original_assignment: Assignment,
                                               bundles_and_fitness: dict[Bundle, float],
                                               num_bidding_jobs: int) -> pd.DataFrame:
    """
    Solve the "maximum-k subset set partitioning problem" and extract bundles from best found solutions.
    Returns a bidding job dataframe in which every carrier has to bid on every bundle

    :param instance:
    :param solution:
    :param original_assignment:
    :param bundles_and_fitness:
    :param num_bidding_jobs:
    :return:
    """
    # TODO how would this work if we used a carrier-aware fitness function?! Then we could solve the WDP directly.
    # prepare data
    bundles, fitness = zip(*bundles_and_fitness.items())
    coef = {b_idx: f for b_idx, f in enumerate(fitness)}
    A = {r: {b_idx: 1 if r in b.requests else 0 for b_idx, b in enumerate(bundles)}
         for r in original_assignment.requests()}

    # 1. create a set partitioning problem
    model: gp.Model
    with gp.Model(
            "Maximum k subset Set Partitioning Problem or Exact Set Covering Problem with at most k elements") as model:
        x: gp.tupledict = model.addVars(range(len(bundles)), vtype=GRB.BINARY, name='x')
        model.setObjective(x.prod(coef), GRB.MAXIMIZE)  # TODO: maximize or minimize?!

        model.setParam(GRB.Param.JSONSolDetail, 1)  # more detail in the json solution
        model.setParam(GRB.Param.PoolSearchMode, 2)  # focus on finding n=PoolSolutions best solutions
        model.setParam(GRB.Param.PoolSolutions,
                       num_bidding_jobs / 2)  # search for num_bidding_jobs solutions, number is only a guess but i don't need num_bididng_jobs solutions and num_bididng_jobs/3 is too few probably
        model.setParam(GRB.Param.MIPFocus, 1)  # focus on finding feasible solutions quickly

        # constraints: all requests are covered exactly once
        for r in original_assignment.requests():
            model.addLConstr(x.prod(A[r]), GRB.EQUAL, 1, f'request {r} is covered')

        # constraints: no more than k bundles are selected
        model.addLConstr(x.sum(), GRB.LESS_EQUAL, instance.num_carriers,
                         f'no more than {instance.num_carriers} bundles are selected')

        model.optimize()
        assert model.SolCount >= num_bidding_jobs, f'Only {model.SolCount} solutions found, but {num_bidding_jobs} ' \
                                                   f'were requested'
        w = []
        z = []
        partitions = []
        for i in range(model.SolCount):
            model.Params.SolutionNumber = i
            w.append(model.Xn)
            z.append(model.PoolObjVal)
            partitions.append([bundles[b_idx] for b_idx, x in enumerate(model.Xn) if x > 0.5])

    # 2. extract bundles from original assignment and the best found solutions
    bundle_extracted = list(original_assignment.bundles())
    for partition in partitions:
        for bundle in partition:
            if bundle not in bundle_extracted:
                bundle_extracted.append(bundle)
        if len(bundle_extracted) >= num_bidding_jobs:
            break

    bidding_jobs_df = pd.DataFrame(
        data=np.full((len(bundle_extracted), instance.num_carriers), True),
        index=bundle_extracted,
        columns=solution.carriers)
    assert bidding_jobs_df.index.is_unique
    return bidding_jobs_df


def bundles_to_bidding_job_df_wdp(instance: it.CAHDInstance,
                                  solution: slt.CAHDSolution,
                                  original_assignment: Assignment,
                                  bundles_and_fitness: dict[Bundle, Sequence[float]],
                                  num_bidding_jobs: int) -> pd.DataFrame:
    bundles, fitness = zip(*bundles_and_fitness.items())
    coef = dict()
    for b_idx, b in enumerate(bundles):
        for carrier in range(instance.num_carriers):
            coef[(b_idx, carrier)] = fitness[b_idx][carrier]

    model: gp.Model
    with gp.Model(name="WDP") as model:
        model.setParam(GRB.Param.OutputFlag, 0)  # suppress output
        model.setParam(GRB.Param.JSONSolDetail, 1)  # more detail in the json solution
        model.setParam(GRB.Param.PoolSearchMode, 2)  # focus on finding n=PoolSolutions best solutions
        model.setParam(GRB.Param.PoolSolutions,
                       num_bidding_jobs / 2)  # search for num_bidding_jobs solutions, number is only a guess but i don't need num_bididng_jobs solutions and num_bididng_jobs/3 is too few probably
        model.setParam(GRB.Param.MIPFocus, 1)  # focus on finding feasible solutions quickly

        y: gp.tupledict = model.addVars(len(bundles_and_fitness), instance.num_carriers, vtype=GRB.BINARY, name='y')
        # since the fitness is maximized, fitness classes such as TrueBids and PreTrainedANN return negative values.
        # Thus, we maximize here the negative fitness, which is equivalent to minimizing the WDP objective function
        model.setObjective(y.prod(coef), GRB.MAXIMIZE)

        # all requests must be covered exactly once
        for r in original_assignment.requests():
            expr = gp.LinExpr()
            for bundle_idx, b in enumerate(bundles):
                if r in b.requests:
                    expr += y.sum(bundle_idx, '*')
            model.addLConstr(expr, GRB.EQUAL, 1, f'request {r} is covered')

        # no carrier can win more than one bundle
        for carrier in range(instance.num_carriers):
            model.addLConstr(y.sum('*', carrier), GRB.LESS_EQUAL, 1, f'carrier {carrier} wins at most one bundle')

        # model.update()
        # model.write('wdp_check.lp')
        model.optimize()

        # extract the best found solutions
        assignments_extracted = []

        if model.Status == GRB.OPTIMAL:
            for i in range(model.SolCount):
                model.Params.SolutionNumber = i
                ass = Assignment(solution.carriers)

                for var_idx, var in enumerate(model.Xn):
                    if var > 0.5:
                        bundle_idx, carrier_idx = var_idx // instance.num_carriers, var_idx % instance.num_carriers
                        for request in bundles[bundle_idx].requests:
                            ass[request] = solution.carriers[carrier_idx]

                assignments_extracted.append((ass, model.PoolObjVal))

    # extract bidding jobs from original assignment and the best found solutions
    combined = sorted(assignments_extracted, key=lambda x: x[1], reverse=True)  # max fitness at the top
    assignments_sorted, fitness_sorted = zip(*combined)
    assignments_sorted = (original_assignment,) + assignments_sorted
    bidding_jobs_df = assignments_to_bidding_job_df(instance, solution, assignments_sorted, num_bidding_jobs)
    return bidding_jobs_df
