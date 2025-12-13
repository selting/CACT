import time
import warnings
from pprint import pprint

import gurobipy as gp
import numpy as np
import pyomo.environ as pyo
import torch
from pyomo.opt import SolverFactory
from sklearn.linear_model import Ridge

from auction_module.bundle_generation.bundle_based.bundle import Bundle

# device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class WdpGurobi:
    def __init__(
        self,
        bundles: list[list[Bundle]],
        valuations: list[list[float]],
        sense: str,
    ):
        self.num_items = len(bundles[0][0].all_items)
        self.num_bidders = len(valuations)
        self.model = None
        self.vars = None
        self.solution = None
        self.solved = False
        self.runtime = None
        self.objVal = None
        self.winner_bids = None
        self.item_allocation = None
        self.sense = gp.GRB.MAXIMIZE if sense == "max" else gp.GRB.MINIMIZE
        self._setup_and_solve(bundles, valuations)

    def _setup_and_solve(
        self, bundles: list[list[Bundle]], valuations: list[list[float]]
    ):
        bundles, valuations = self._drop_infeasible(bundles, valuations)
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env, name="WDP") as self.model:
                # variables, sparse
                vars_a = []
                for bidder_idx in range(self.num_bidders):
                    for bundle_idx in range(len(bundles[bidder_idx])):
                        vars_a.append((bidder_idx, bundle_idx))
                self.vars = self.model.addVars(vars_a, vtype=gp.GRB.BINARY, name=f"a_")

                # objective
                objective = []
                for bidder_idx in range(self.num_bidders):
                    for bundle_idx in range(len(bundles[bidder_idx])):
                        objective.append(
                            self.vars[bidder_idx, bundle_idx]
                            * valuations[bidder_idx][bundle_idx]
                        )
                self.model.setObjective(gp.quicksum(objective), sense=self.sense)

                # constraints
                # each bidder wins at most one bundle
                for bidder_idx in range(self.num_bidders):
                    lhs = []
                    for bundle_idx in range(len(bundles[bidder_idx])):
                        lhs.append(self.vars[bidder_idx, bundle_idx])
                    self.model.addConstr(
                        gp.quicksum(lhs) <= 1, name=f"bidder_{bidder_idx}"
                    )

                # each item is allocated to (a) at most or (b) exactly one bidder
                for item_idx in range(self.num_items):
                    lhs = []
                    for bidder_idx in range(self.num_bidders):
                        for bundle_idx in range(len(bundles[bidder_idx])):
                            if bundles[bidder_idx][bundle_idx].bitstring[item_idx] in (
                                "1",
                                1,
                                True,
                            ):
                                lhs.append(self.vars[bidder_idx, bundle_idx])
                    self.model.addConstr(
                        gp.quicksum(lhs) == 1, name=f"item_{item_idx}"
                    )  # (a) <= 1, (b) == 1

                # SOLVE
                if self.solved:
                    raise RuntimeError("Model already solved!")
                self.model.optimize()
                if not self.model.status == gp.GRB.OPTIMAL:
                    raise ValueError("Optimization failed.")
                self.solved = True
                # get the optimal allocation
                solution = [
                    [self.vars[k, b].x for b in range(len(bundles[k]))]
                    for k in range(self.num_bidders)
                ]
                item_allocation = np.zeros(
                    (self.num_bidders, self.num_items), dtype=np.int32
                )
                winner_bids: list = [None for _ in range(self.num_bidders)]
                for bidder_idx in range(self.num_bidders):
                    for bundle_idx in range(len(bundles[bidder_idx])):
                        if solution[bidder_idx][bundle_idx] > 0.5:
                            item_allocation[bidder_idx] = bundles[bidder_idx][
                                bundle_idx
                            ].bitstring
                            winner_bids[bidder_idx] = valuations[bidder_idx][bundle_idx]
                            break
                self.solution = solution
                self.item_allocation = [tuple(x) for x in item_allocation]
                self.winner_bids = winner_bids
                self.objVal = self.model.objVal
                self.runtime = self.model.Runtime
        pass

    def print_solution(self):
        print("Solution:")
        pprint(self.item_allocation)
        print(f"Optimal value: {self.objVal}")
        print(f"Runtime: {self.runtime:.2f}s")

    def _drop_infeasible(self, bundles: list[list[Bundle]], valuations):
        """
        filter out infeasible bundles and their corresponding valuations.

        :param bundles:
        :param valuations:
        :return:
        """
        feasible_bundles = []
        feasible_valuations = []
        for bidder_idx in range(self.num_bidders):
            feasible_bundles_bidder = []
            feasible_valuations_bidder = []
            for bundle_idx in range(len(bundles[bidder_idx])):
                bundle = bundles[bidder_idx][bundle_idx]
                valuation = valuations[bidder_idx][bundle_idx]
                if valuation == "infeasible":
                    continue
                else:
                    feasible_bundles_bidder.append(bundle)
                    feasible_valuations_bidder.append(valuation)
            feasible_bundles.append(feasible_bundles_bidder)
            feasible_valuations.append(feasible_valuations_bidder)
        return feasible_bundles, feasible_valuations


# class WdpPyomo:
#     def __init__(self,
#                  bundles: list[list[Bundle]],
#                  valuations: list[list[float]],
#                  sense: str,
#                  solver_name: str = 'gurobi'  # specify solver name, default to gurobi
#                  ):
#         self.num_items = len(bundles[0][0].all_items)
#         self.num_bidders = len(valuations)
#         self.model = None
#         self.vars = None
#         self.solution = None
#         self.solved = False
#         self.runtime = None
#         self.objVal = None
#         self.winner_bids = None
#         self.item_allocation = None
#         self.sense = pyo.maximize if sense == 'max' else pyo.minimize
#         self.solver_name = solver_name  # store solver name
#         self._setup_and_solve(bundles, valuations)
#
#     def _setup_and_solve(self, bundles: list[list[Bundle]], valuations: list[list[float]]):
#         bundles, valuations = self._drop_infeasible(bundles, valuations)
#         model = pyo.ConcreteModel()
#
#         # Index sets
#         model.N = pyo.RangeSet(self.num_bidders)
#         model.M = pyo.RangeSet(self.num_items)
#
#         # Generate index set for bidder-bundle combinations
#         combinations_list = []
#         bid_values = {}
#         bundle_items = {}
#
#         for bidder_index, bidder in enumerate(range(self.num_bidders)):
#             for bundle_index, bundle in enumerate(bundles[bidder_index]):
#                 # Create a unique bundle identifier for each bidder
#                 combination_id = (bidder, f"bundle{bundle_index + 1}")
#                 combinations_list.append(combination_id)
#                 bid_values[combination_id] = valuations[bidder_index][bundle_index]
#                 bundle_items[combination_id] = bundle  # requires hashability in Pyomo
#
#         model.combinations = pyo.Set(initialize=combinations_list, dimen=2)
#
#         # Parameters
#         model.bids = pyo.Param(model.combinations, initialize=bid_values)
#         model.bundles = pyo.Param(model.combinations, initialize=bundle_items, within=pyo.Any)
#
#         # Variables   todo: verify that the variables are correct
#         model.x = pyo.Var(model.combinations, domain=pyo.Binary)
#
#         # Objective function
#         def obj_rule(model):
#             return sum(model.bids[c] * model.x[c] for c in model.combinations)
#
#         model.obj = pyo.Objective(rule=obj_rule, sense=self.sense)
#
#         # Constraint 1: Each item must be assigned to exactly one bidder
#         @pyo.simple_constraint_rule
#         def item_coverage_rule(model, item_idx):
#             return sum(model.x[c] for c in model.combinations
#                        if model.bundles[c].bitstring[item_idx - 1] in ('1', 1, True)) == 1
#
#         model.item_coverage_constraint = pyo.Constraint(model.M, rule=item_coverage_rule)
#
#         # Constraint 2: At most one combination per bidder
#         @pyo.simple_constraint_rule
#         def bidder_bundle_limit_rule(model, bidder):
#             return sum(model.x[c] for c in model.combinations if c[0] == bidder) <= 1
#
#         model.bidder_bundle_limit_constraint = pyo.Constraint(model.N, rule=bidder_bundle_limit_rule)
#
#         # SOLVE
#         if self.solved:
#             raise RuntimeError('Model already solved!')
#         solver = SolverFactory(self.solver_name)
#         start_time = time.time()
#         results = solver.solve(model, tee=False)  # tee=True to see solver output
#         end_time = time.time()
#         self.runtime = end_time - start_time
#
#         if not (results.solver.status == pyo.SolverStatus.ok and results.solver.termination_condition == pyo.TerminationCondition.optimal):
#             raise ValueError('Optimization failed.')
#         else:
#             self.model = model
#             self.vars = model.x
#             self.solved = True
#             self.objVal = pyo.value(model.obj)
#
#             # Get the optimal allocation
#             solution = [pyo.value(model.x[c]) for c in model.combinations]
#             item_allocation = np.zeros((self.num_bidders, self.num_items), dtype=np.int32)
#             winner_bids: list = [None for _ in range(self.num_bidders)]
#
#             for combination in model.combinations:
#                 if model.x[combination].value > 0.5:
#                     item_allocation[combination[0]] = model.bundles[combination].bitstring
#                     winner_bids[combination[0]] = model.bids[combination]
#             if not np.all(item_allocation.sum(axis=0) == 1):
#                 raise ConstraintViolationError('Constraint violation: items go assigned more than once or did not get'
#                                                'assigned at all.')
#             self.solution = solution
#             self.item_allocation = [tuple(x) for x in item_allocation]
#             self.winner_bids = winner_bids
#
#     def print_solution(self):
#         print('Solution:')
#         pprint(self.item_allocation)
#         print(f'Optimal value: {self.objVal}')
#         print(f'Runtime: {self.runtime:.2f}s')
#
#     def _drop_infeasible(self, bundles: list[list[Bundle]], valuations):
#         """
#         filter out infeasible bundles and their corresponding valuations.
#
#         :param bundles:
#         :param valuations:
#         :return:
#         """
#         feasible_bundles = []
#         feasible_valuations = []
#         for bidder_idx in range(self.num_bidders):
#             feasible_bundles_bidder = []
#             feasible_valuations_bidder = []
#             for bundle_idx in range(len(bundles[bidder_idx])):
#                 bundle = bundles[bidder_idx][bundle_idx]
#                 valuation = valuations[bidder_idx][bundle_idx]
#                 if valuation == 'infeasible':
#                     continue
#                 else:
#                     feasible_bundles_bidder.append(bundle)
#                     feasible_valuations_bidder.append(valuation)
#             feasible_bundles.append(feasible_bundles_bidder)
#             feasible_valuations.append(feasible_valuations_bidder)
#         return feasible_bundles, feasible_valuations


class WdpPyomo:
    def __init__(
        self,
        bundles: list[list["Bundle"]],
        valuations: list[list[float]],
        sense: str,
        solver_name: str = "glpk",
        # default solver is glpk, can be changed to 'cbc', 'gurobi' etc. if installed
    ):
        self.num_items = len(bundles[0][0].all_items)
        self.num_bidders = len(valuations)
        self.model = None
        self.vars = None
        self.solution = None
        self.solved = False
        self.runtime = None
        self.objVal = None
        self.winner_bids = None
        self.item_allocation = None
        self.sense = (
            pyo.minimize if sense == "min" else pyo.maximize
        )  # Pyomo uses minimize/maximize, not GRB.MINIMIZE/MAXIMIZE
        self.solver_name = solver_name
        self._setup_and_solve(bundles, valuations)

    def _setup_and_solve(
        self, bundles: list[list[Bundle]], valuations: list[list[float]]
    ):
        bundles, valuations = self._drop_infeasible(bundles, valuations)
        self.model = pyo.ConcreteModel()

        # Variables
        vars_a = []
        for bidder_idx in range(self.num_bidders):
            for bundle_idx in range(len(bundles[bidder_idx])):
                vars_a.append((bidder_idx, bundle_idx))
        self.model.vars = pyo.Var(vars_a, domain=pyo.Binary, name="a")

        # Objective
        objective_expr = sum(
            self.model.vars[bidder_idx, bundle_idx] * valuations[bidder_idx][bundle_idx]
            for bidder_idx in range(self.num_bidders)
            for bundle_idx in range(len(bundles[bidder_idx]))
        )
        self.model.objective = pyo.Objective(expr=objective_expr, sense=self.sense)

        # Constraints
        self.model.bidder_constraints = pyo.ConstraintList()
        for bidder_idx in range(self.num_bidders):
            lhs = sum(
                self.model.vars[bidder_idx, bundle_idx]
                for bundle_idx in range(len(bundles[bidder_idx]))
            )
            self.model.bidder_constraints.add(lhs <= 1)

        self.model.item_constraints = pyo.ConstraintList()
        for item_idx in range(self.num_items):
            lhs = sum(
                self.model.vars[bidder_idx, bundle_idx]
                for bidder_idx in range(self.num_bidders)
                for bundle_idx in range(len(bundles[bidder_idx]))
                if bundles[bidder_idx][bundle_idx].bitstring[item_idx] == 1
            )
            self.model.item_constraints.add(lhs == 1)

        # Solve
        if self.solved:
            raise RuntimeError("Model already solved!")

        solver = SolverFactory(self.solver_name)  # Use the specified solver
        if not solver.available():
            raise ValueError(
                f"Solver {self.solver_name} is not available. Please install it or choose a different solver."
            )

        start_time = time.time()
        solution = solver.solve(
            self.model
        )  # , tee=True)  # , timelimit=None) # you can set a timelimit here if needed
        self.runtime = time.time() - start_time

        if solution.solver.termination_condition != pyo.TerminationCondition.optimal:
            raise ValueError(
                f"Optimization failed. Termination condition: {solution.solver.termination_condition}"
            )
        self.solved = True

        # Get the optimal allocation
        pyomo_solution = [
            [pyo.value(self.model.vars[k, b]) for b in range(len(bundles[k]))]
            for k in range(self.num_bidders)
        ]  # Access solution values using value()
        item_allocation = np.zeros((self.num_bidders, self.num_items), dtype=np.int32)
        winner_bids: list = [None for _ in range(self.num_bidders)]
        for bidder_idx in range(self.num_bidders):
            for bundle_idx in range(len(bundles[bidder_idx])):
                if pyomo_solution[bidder_idx][bundle_idx] > 0.5:
                    item_allocation[bidder_idx] = bundles[bidder_idx][
                        bundle_idx
                    ].bitstring
                    winner_bids[bidder_idx] = valuations[bidder_idx][bundle_idx]
                    break
        self.solution = pyomo_solution
        self.item_allocation = [tuple(x) for x in item_allocation]
        self.winner_bids = winner_bids
        self.objVal = pyo.value(
            self.model.objective
        )  # Access objective value using value()

    def print_solution(self):
        print("Solution:")
        pprint(self.item_allocation)
        print(f"Optimal value: {self.objVal}")
        print(f"Runtime: {self.runtime:.2f}s")

    def _drop_infeasible(self, bundles: list[list["Bundle"]], valuations):
        """
        filter out infeasible bundles and their corresponding valuations.

        :param bundles:
        :param valuations:
        :return:
        """
        feasible_bundles = []
        feasible_valuations = []
        for bidder_idx in range(self.num_bidders):
            feasible_bundles_bidder = []
            feasible_valuations_bidder = []
            for bundle_idx in range(len(bundles[bidder_idx])):
                bundle = bundles[bidder_idx][bundle_idx]
                valuation = valuations[bidder_idx][bundle_idx]
                if valuation == "infeasible":
                    continue
                else:
                    feasible_bundles_bidder.append(bundle)
                    feasible_valuations_bidder.append(valuation)
            feasible_bundles.append(feasible_bundles_bidder)
            feasible_valuations.append(feasible_valuations_bidder)
        return feasible_bundles, feasible_valuations


class WdpAllSolutions:
    def __init__(
        self,
        bundles: list[list[tuple[int]]],
        valuations: list[list[float]],
        verbose=0,
        sense="max",
    ):
        self.num_items = len(bundles[0][0])
        self.num_bidders = len(valuations)
        self.model = None
        self.vars = None
        self.solution = None
        self.solved = False
        self.runtime = None
        self.objVal = None
        self.winner_bids = None
        self.item_allocation = None
        self.sense = gp.GRB.MAXIMIZE if sense == "max" else gp.GRB.MINIMIZE
        self.setup_and_solve(bundles, valuations, verbose)

    def setup_and_solve(
        self, bundles: list[list[tuple[int]]], valuations: list[list[float]], verbose=0
    ):
        bundles, valuations = self.drop_infeasible(bundles, valuations)
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env, name="WDP") as self.model:
                # set params to find all solutions
                self.model.Params.PoolSearchMode = (
                    2  # find the k=Params.PoolSolution best solutions
                )
                self.model.Params.PoolSolutions = gp.GRB.MAXINT  # find all solutions

                # variables, sparse
                vars_a = []
                for bidder_idx in range(self.num_bidders):
                    for bundle_idx in range(len(bundles[bidder_idx])):
                        vars_a.append((bidder_idx, bundle_idx))
                self.vars = self.model.addVars(vars_a, vtype=gp.GRB.BINARY, name=f"a_")

                # objective
                objective = []
                for bidder_idx in range(self.num_bidders):
                    for bundle_idx in range(len(bundles[bidder_idx])):
                        objective.append(
                            self.vars[bidder_idx, bundle_idx]
                            * valuations[bidder_idx][bundle_idx]
                        )
                self.model.setObjective(gp.quicksum(objective), sense=self.sense)

                # constraints
                # each bidder wins at most one bundle
                for bidder_idx in range(self.num_bidders):
                    lhs = []
                    for bundle_idx in range(len(bundles[bidder_idx])):
                        lhs.append(self.vars[bidder_idx, bundle_idx])
                    self.model.addConstr(
                        gp.quicksum(lhs) <= 1, name=f"bidder_{bidder_idx}"
                    )

                # each item is allocated to (a) at most or (b) exactly one bidder
                for item_idx in range(self.num_items):
                    lhs = []
                    for bidder_idx in range(self.num_bidders):
                        for bundle_idx in range(len(bundles[bidder_idx])):
                            if bundles[bidder_idx][bundle_idx][item_idx] in (
                                "1",
                                1,
                                True,
                            ):
                                lhs.append(self.vars[bidder_idx, bundle_idx])
                    self.model.addConstr(
                        gp.quicksum(lhs) == 1, name=f"item_{item_idx}"
                    )  # (a) <= 1, (b) == 1

                # SOLVE
                assert self.solved is False
                self.model.optimize()
                assert self.model.status == gp.GRB.OPTIMAL
                self.solved = True
                # get all the allocations
                self.solution = []
                self.item_allocation = []
                self.winner_bids = []
                self.objVal = []
                for solution_idx in range(self.model.SolCount):
                    self.model.Params.SolutionNumber = solution_idx
                    solution = [
                        [self.vars[k, b].Xn for b in range(len(bundles[k]))]
                        for k in range(self.num_bidders)
                    ]
                    item_allocation = np.zeros(
                        (self.num_bidders, self.num_items), dtype=np.int32
                    )
                    winner_bids: list = [None for _ in range(self.num_bidders)]
                    for bidder_idx in range(self.num_bidders):
                        for bundle_idx in range(len(bundles[bidder_idx])):
                            if solution[bidder_idx][bundle_idx] > 0.5:
                                item_allocation[bidder_idx] = bundles[bidder_idx][
                                    bundle_idx
                                ]
                                winner_bids[bidder_idx] = valuations[bidder_idx][
                                    bundle_idx
                                ]
                                break
                    self.solution.append(solution)
                    self.item_allocation.append(
                        tuple(tuple(x) for x in item_allocation)
                    )
                    self.winner_bids.append(winner_bids)
                    self.objVal.append(self.model.PoolObjVal)
                # ----------------
                self.runtime = self.model.Runtime
        pass

    def print_solution(self):
        print("Solution:")
        pprint(self.item_allocation)
        print(f"Optimal value: {self.objVal}")
        print(f"Runtime: {self.runtime:.2f}s")

    def drop_infeasible(self, bundles, valuations):
        """
        filter out infeasible bundles and their corresponding valuations.

        :param bundles:
        :param valuations:
        :return:
        """
        feasible_bundles = []
        feasible_valuations = []
        for bidder_idx in range(self.num_bidders):
            feasible_bundles_bidder = []
            feasible_valuations_bidder = []
            for bundle_idx in range(len(bundles[bidder_idx])):
                bundle = bundles[bidder_idx][bundle_idx]
                valuation = valuations[bidder_idx][bundle_idx]
                if valuation == "infeasible":
                    continue
                else:
                    feasible_bundles_bidder.append(bundle)
                    feasible_valuations_bidder.append(valuation)
            feasible_bundles.append(feasible_bundles_bidder)
            feasible_valuations.append(feasible_valuations_bidder)
        return feasible_bundles, feasible_valuations


class WDPRidgeRegression:
    def __init__(
        self,
        ridge_regression_models: list[Ridge],
        sense: str,
        blacklist_bundles=False,
        carrier_needs_query_constraints: list[bool] = False,
        gurobi_num_threads=gp.GRB.Param.Threads,  # default 0 -> use all available cores
    ):
        """

        Parameters
        ----------
        ridge_regression_models: A list of sklearn Ridge regression models, one for each bidder.
        sense: str: 'max' or 'min', whether to maximize or minimize the objective function.
        blacklist_bundles: Optional: a list of num_carriers lists of bundles to exclude from the set of possible solutions.
        carrier_needs_query_constraints: Optional: whether to add constraints to ensure that some specified carriers
         receive at least one query. list of bools, one for each carrier.
        gurobi_num_threads: Optional: the number of threads to use for the Gurobi solver. Default is 0, which uses all
            available cores.
        """

        self.num_items = len(ridge_regression_models[0].coef_)
        self.num_bidders = len(ridge_regression_models)
        self.model = None
        self.vars: gp.tupledict = None
        self.solution = None
        self.solved = False
        self.runtime = None
        self.objVal = None
        self.winner_bids = None
        self.item_allocation = None
        self.sense = gp.GRB.MAXIMIZE if sense == "max" else gp.GRB.MINIMIZE
        self.blacklist_bundles = blacklist_bundles
        self.carrier_needs_query_constraints = carrier_needs_query_constraints
        self.gurobi_num_threads = gurobi_num_threads

        # self.vars_diff = None  # aux variables
        # self.vars_abs = None  # aux variables

        self.setup_and_solve(ridge_regression_models)

    def setup_and_solve(self, ridge_regression_models):
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model("WDPRegularizedLinearRegression") as self.model:
                self.model.Params.OutputFlag = 0
                self.model.Params.Threads = self.gurobi_num_threads
                # variables: should bidder i get item j
                self.vars = self.model.addVars(
                    self.num_bidders, self.num_items, vtype=gp.GRB.BINARY, name="a"
                )

                # objective
                objective = []
                for i in range(self.num_bidders):
                    w_i = ridge_regression_models[i].coef_
                    for j in range(self.num_items):
                        objective.append(w_i[j] * self.vars[i, j])
                self.model.setObjective(gp.quicksum(objective), sense=gp.GRB.MAXIMIZE)

                # constraints
                # every item is assigned to (a) at most one bidder or (b) exactly one bidder
                # (depends on the free disposal assumption)
                for j in range(self.num_items):
                    self.model.addConstr(
                        self.vars.sum("*", j) == 1, name=f"item_{j}"
                    )  # (b) == 1
                    # self.model.addConstr(self.vars.sum('*', j) <= 1, name=f'item_{j}')  # (a) <= 1

                if self.blacklist_bundles:
                    # exclude blacklisted bundles by adding integer cuts
                    for carrier_idx, carrier_blacklist in enumerate(
                        self.blacklist_bundles
                    ):
                        for bundle in carrier_blacklist:
                            expr = []
                            for j in range(self.num_items):
                                expr.append(
                                    (self.vars[carrier_idx, j] - bundle[j]) ** 2
                                )
                                # TODO ideally, this would just use the absolute value and thus avoid quadratic terms.
                                #  However, I believe this would require auxiliary variables, making it difficult to
                                #  implement.
                            self.model.addConstr(
                                gp.quicksum(expr) >= 1, name=f"blacklist_{carrier_idx}"
                            )

                if any(self.carrier_needs_query_constraints):
                    # add constraints to ensure that some specified carriers receive at least one query
                    for carrier_idx, needs_query in enumerate(
                        self.carrier_needs_query_constraints
                    ):
                        if needs_query:
                            self.model.addConstr(
                                self.vars.sum(carrier_idx, "*") >= 1,
                                name=f"needs_query_{carrier_idx}",
                            )

                # SOLVE
                assert self.solved is False
                self.model.optimize()
                assert self.model.status == gp.GRB.OPTIMAL
                self.solved = True
                # get the optimal allocation
                self.solution = np.array(
                    [
                        [self.vars[i, j].x for i in range(self.num_bidders)]
                        for j in range(self.num_items)
                    ]
                )
                item_allocation = np.array((self.solution > 0.5).T, dtype=np.int32)
                self.item_allocation = [tuple(x) for x in item_allocation]
                # TODO self.item_allocation is usually a list of tuples

                self.winner_bids = [
                    None for _ in range(self.num_bidders)
                ]  # there are no true bids with Ridge WDP
                self.objVal = self.model.objVal
                self.runtime = self.model.Runtime
        pass


class WDPNeuralNet:
    def __init__(
        self,
        nn_models,
        bigM,
        tighten_bounds,
        blacklist_bundles: dict[int, torch.Tensor] = None,
        num_solutions=1,
        free_disposal=False,
        starts=None,
        verbose=0,
        **kwargs,
    ):
        """

        :param nn_models: List of Neural Network model to use for the WDP
        :param bigM: initial upper bound for the variables z and s
        :param tighten_bounds: whether to tighten the upper bounds for the variables z and s using interval arithmetic
        :param blacklist_bundles: set of bundles to exclude from the set of possible solutions (not working as of now)
        :param num_solutions: the number of ordered top solutions to find and return
        :param free_disposal: whether to assume free disposal (not all items must be assigned) or not (all must be assigned)
        :param starts: list of initial feasible solutions to start the search from
        :param verbose: whether to print the output of the solver or not
        """
        self.ml_models = nn_models
        self.tighten_bounds = tighten_bounds
        self.blacklist_bundles = blacklist_bundles
        self.num_solutions = num_solutions
        self.free_disposal = free_disposal
        self.starts = starts

        self.num_bidders = len(nn_models)
        self.num_items = nn_models[0].input_dim
        self.verbose = verbose

        self.mip = None
        self.vars_y: gp.tupledict = gp.tupledict()
        self.vars_z: gp.tupledict = gp.tupledict()
        self.vars_s: gp.tupledict = gp.tupledict()

        self.bigM = bigM
        self.upper_bounds_z = dict()  # defined with "combined_indices"
        self.upper_bounds_s = dict()  # defined with "combined_indices"
        for bidder_idx in range(self.num_bidders):
            weights_and_biases = list(self.ml_models[bidder_idx].parameters())
            comb_layer_idx = 1
            for v in range(0, len(weights_and_biases), 2):
                num_out_feat, num_in_feat = weights_and_biases[v].shape
                for out_feat in range(num_out_feat):
                    self.upper_bounds_z[bidder_idx, comb_layer_idx, out_feat] = (
                        self.bigM
                    )
                    self.upper_bounds_s[bidder_idx, comb_layer_idx, out_feat] = (
                        self.bigM
                    )
                comb_layer_idx += 1

        self.solved = False
        self.solutions = []
        self.item_allocations = []
        self.objVals = []
        self.runtime = None
        self.setup_and_solve(
            tighten_bounds=self.tighten_bounds, blacklist_bundles=self.blacklist_bundles
        )

    def setup_and_solve(self, tighten_bounds=False, blacklist_bundles=None):
        if blacklist_bundles is not None:
            raise NotImplementedError(
                "Blacklisting bundles is not working as intended!"
            )
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env, name="WDPNeuralNet") as self.mip:
                self._setup()
                if self.starts:
                    self._set_starts()
                if tighten_bounds:
                    self._tighten_bounds_IA(upper_bound=self.bigM)
                self._solve()
                # print(f'Number of variables: {self.mip.NumVars}, Number of constraints: {self.mip.NumConstrs},'
                #       f' Number of nonzeros: {self.mip.NumNZs}, Runtime: {self.runtime:.2f}s')
        pass

    def _setup(self):
        self.mip.Params.OutputFlag = self.verbose
        # constraints
        # TODO z_i_o == a_i  ??? (note that a_i is not a variable but the bundle of bidder i in allocation a!)
        for bidder_idx in range(self.num_bidders):
            self._add_matrix_constraints(bidder_idx)
        # every item is assigned to (a) at most or (b) exactly one bidder [depending on free disposal assumption]
        for j in range(self.num_items):
            lhs = gp.quicksum(
                [
                    self.vars_z[bidder_idx, 0, j]
                    for bidder_idx in range(self.num_bidders)
                ]
            )
            if self.free_disposal:
                self.mip.addConstr(lhs <= 1, name=f"item_allocation_{j}")  # (a) <= 1
            else:
                self.mip.addConstr(lhs == 1, name=f"item_allocation_{j}")  # (b) == 1
        # objective
        objective = []
        for bidder_idx in range(self.num_bidders):
            output_layer_idx = (
                len(self.ml_models[bidder_idx].layers) // 2
            )  # TODO on which index basis is z defined?
            objective.append(
                self.vars_z[bidder_idx, output_layer_idx, 0]
            )  # 0 because output has only one node
        self.mip.setObjective(gp.quicksum(objective), sense=gp.GRB.MAXIMIZE)
        # exclude blacklisted bundles
        # if self.blacklist_bundles:
        #     self._blacklist_bundles()

    def _set_starts(self):
        self.starts: list
        self.mip.NumStart = len(self.starts)
        for start_idx, start in enumerate(self.starts):
            self.mip.Params.StartNumber = start_idx
            for bidder_idx in range(self.num_bidders):
                for item_idx in range(self.num_items):
                    self.vars_z[bidder_idx, 0, item_idx].Start = start[bidder_idx][
                        item_idx
                    ]
        pass

    def _solve(self):
        assert self.solved is False
        # self.mip.update()
        # self.mip.write('WDPNeuralNet.lp')
        if self.num_solutions > 1:
            self.mip.Params.PoolSearchMode = (
                2  # find the k=Params.PoolSolution best solutions
            )
            self.mip.Params.PoolSolutions = self.num_solutions
        # TODO provide a warmstart solution
        self.mip.optimize()
        assert self.mip.status == gp.GRB.OPTIMAL
        self.solved = True
        # get the optimal allocations
        for solution_idx in range(self.mip.SolCount):
            self.mip.Params.SolutionNumber = solution_idx
            solution = np.array(
                [
                    [self.vars_z[i, 0, j].Xn for i in range(self.num_bidders)]
                    for j in range(self.num_items)
                ]
            )
            item_allocation = (solution > 0.5).T.astype(np.int32)
            self.solutions.append(solution)
            self.item_allocations.append(item_allocation)
            self.objVals.append(self.mip.PoolObjVal)
        self.runtime = self.mip.Runtime

    def _tighten_bounds_IA(self, upper_bound):
        """
        Tighten the upper bounds for the variables z and s using interval arithmetic.
        :param upper_bound:
        :return:
        """
        warnings.warn(
            "Implementation of tightening bounds using interval arithmetic has not been thoroughly tested."
        )
        for bidder_idx in range(self.num_bidders):
            weights_and_biases = list(self.ml_models[bidder_idx].parameters())
            comb_layer_idx = 0
            for j in range(0, len(weights_and_biases), 2):
                weight = weights_and_biases[j]
                bias = weights_and_biases[j + 1]
                num_out_features, num_in_features = weight.shape
                if j == 0:
                    for out_feat in range(
                        num_out_features
                    ):  # in GitHub repo it loops over in_feat for 0th layer
                        self.upper_bounds_z[bidder_idx, 0, out_feat] = upper_bound
                        self.upper_bounds_s[bidder_idx, 0, out_feat] = upper_bound
                else:
                    w_plus = torch.maximum(weight, torch.zeros_like(weight))
                    w_minus = torch.minimum(weight, torch.zeros_like(weight))
                    for out_feat in range(num_out_features):
                        X_z = torch.tensor(
                            [
                                self.upper_bounds_z[bidder_idx, comb_layer_idx - 1, i]
                                for i in range(num_in_features)
                            ],
                            device=device,
                            dtype=torch.float32,
                        )
                        ub_z = torch.maximum(
                            w_plus[out_feat, :] @ X_z + bias[out_feat],
                            torch.tensor(0.0, device=device),
                        )
                        if self.verbose:
                            print(
                                f"ub_z: old: {self.upper_bounds_z[bidder_idx, comb_layer_idx, out_feat]}, "
                                f"new: {torch.ceil(ub_z)}\tIncreased: {bool(torch.ceil(ub_z) > self.upper_bounds_z[bidder_idx, comb_layer_idx, out_feat])}"
                            )
                        self.upper_bounds_z[bidder_idx, comb_layer_idx, out_feat] = (
                            torch.ceil(ub_z)
                        )

                        X_s = torch.tensor(
                            [
                                self.upper_bounds_z[bidder_idx, comb_layer_idx - 1, i]
                                for i in range(num_in_features)
                            ],
                            device=device,
                            dtype=torch.float32,
                        )
                        ub_s = torch.maximum(
                            -(w_minus[out_feat, :] @ X_s + bias[out_feat]),
                            torch.tensor(0.0, device=device),
                        )
                        if self.verbose:
                            print(
                                f"ub_s: old: {self.upper_bounds_s[bidder_idx, comb_layer_idx, out_feat]}, "
                                f"new: {torch.ceil(ub_s)}\tIncreased: {bool(torch.ceil(ub_s) > self.upper_bounds_s[bidder_idx, comb_layer_idx, out_feat])}"
                            )
                        self.upper_bounds_s[bidder_idx, comb_layer_idx, out_feat] = (
                            torch.ceil(ub_s)
                        )

                comb_layer_idx += 1
        pass

    def _blacklist_bundles(self):
        """
        Exclude blacklisted bundles of different bidders to be part of the solution.
        """
        raise NotImplementedError(
            "Blacklisting bundles is not working as intended! Could not figure out the problem"
        )
        self.vars_diff = {}
        self.vars_abs = {}
        for bidder_idx, blacklist in self.blacklist_bundles.items():
            # vars for the difference between the bundle and the blacklisted bundle
            self.vars_diff[bidder_idx] = self.mip.addVars(
                len(blacklist),
                self.num_items,
                vtype=gp.GRB.CONTINUOUS,
                name=f"diff_{bidder_idx}",
            )
            self.vars_abs[bidder_idx] = self.mip.addVars(
                len(blacklist),
                self.num_items,
                lb=0,
                vtype=gp.GRB.CONTINUOUS,
                name=f"abs_{bidder_idx}",
            )
            for bundle_idx, bundle in enumerate(blacklist):
                for item_idx in range(self.num_items):
                    self.mip.addConstr(
                        self.vars_diff[bidder_idx][bundle_idx, item_idx]
                        == bundle[item_idx].item()
                        - self.vars_z[bidder_idx, 0, item_idx],
                        name=f"diff_{bidder_idx}_{bundle_idx}_{item_idx}",
                    )

                    self.mip.addGenConstrAbs(
                        self.vars_abs[bidder_idx][bundle_idx, item_idx],
                        self.vars_diff[bidder_idx][bundle_idx, item_idx],
                        name=f"abs_{bidder_idx}_{bundle_idx}_{item_idx}",
                    )

                    # self.mip.addConstr(self.vars_abs[bidder_idx][bundle_idx, item_idx] == gp.abs_(
                    #     self.vars_diff[bidder_idx][bundle_idx, item_idx]),
                    #                    name=f'abs_{bidder_idx}_{bundle_idx}_{item_idx}')

                self.mip.addConstr(
                    gp.quicksum(self.vars_abs[bidder_idx].select(bundle_idx, "*")) >= 1,
                    name=f"blacklist_{bidder_idx}_{bundle_idx}",
                )
        # self.mip.update()
        # self.mip.write('WDPNeuralNet.lp')
        pass

    def _add_matrix_constraints(self, bidder_idx):
        weights_and_biases = list(self.ml_models[bidder_idx].parameters())
        layer_idx = 1
        for v in range(0, len(weights_and_biases), 2):
            weight = weights_and_biases[v]
            bias = weights_and_biases[v + 1]
            # print(f'weight.shape:\t{weight.shape}\nbias.shape:\t\t{bias.shape}')
            num_out_features, num_in_features = weight.shape

            # decision variables
            if v == 0:
                z = [(bidder_idx, 0, in_feat) for in_feat in range(num_in_features)]
                self.vars_z.update(self.mip.addVars(z, vtype=gp.GRB.BINARY, name="a"))

            z = [
                (bidder_idx, layer_idx, out_feat)
                for out_feat in range(num_out_features)
            ]
            self.vars_z.update(
                self.mip.addVars(z, lb=0, vtype=gp.GRB.CONTINUOUS, name="z")
            )

            s = [
                (bidder_idx, layer_idx, out_feat)
                for out_feat in range(num_out_features)
                if (
                    self.upper_bounds_z[bidder_idx, layer_idx, out_feat] != 0
                    and self.upper_bounds_s[bidder_idx, layer_idx, out_feat] != 0
                )
            ]
            self.vars_s.update(
                self.mip.addVars(s, lb=0, vtype=gp.GRB.CONTINUOUS, name="s")
            )

            y = [
                (bidder_idx, layer_idx, out_feat)
                for out_feat in range(num_out_features)
                if (
                    self.upper_bounds_z[bidder_idx, layer_idx, out_feat] != 0
                    and self.upper_bounds_s[bidder_idx, layer_idx, out_feat] != 0
                )
            ]
            self.vars_y.update(self.mip.addVars(y, vtype=gp.GRB.BINARY, name="y"))

            # add constraints
            for out_feat in range(num_out_features):
                # print(f'Row {out_feat}')
                # print(f'W[r, ]: {weight[out_feat, :].detach().cpu().numpy()}')
                # print(f'b[r]: {bias[out_feat]}')
                # print(f'upper z-bound: {self.upper_bounds_z[bidder_idx, layer_idx, out_feat]}')
                # print(f'upper s-bound: {self.upper_bounds_s[bidder_idx, layer_idx, out_feat]}')

                if self.upper_bounds_z[bidder_idx, layer_idx, out_feat] == 0:
                    self.mip.addConstr(
                        self.vars_z[bidder_idx, layer_idx, out_feat] == 0,
                        f"z_{bidder_idx}_{layer_idx}_{out_feat}",
                    )

                elif self.upper_bounds_s[bidder_idx, layer_idx, out_feat] == 0:
                    lhs = (
                        gp.quicksum(
                            weight[out_feat, in_feat]
                            * self.vars_z[bidder_idx, layer_idx - 1, in_feat]
                            for in_feat in range(num_in_features)
                        )
                        + bias[out_feat]
                    )
                    self.mip.addConstr(
                        lhs == self.vars_z[bidder_idx, layer_idx, out_feat],
                        f"z_{bidder_idx}_{layer_idx}_{out_feat}",
                    )

                else:
                    lhs = (
                        gp.quicksum(
                            weight[out_feat, in_feat]
                            * self.vars_z[bidder_idx, layer_idx - 1, in_feat]
                            for in_feat in range(num_in_features)
                        )
                        + bias[out_feat]
                    )
                    self.mip.addConstr(
                        lhs
                        == self.vars_z[bidder_idx, layer_idx, out_feat]
                        - self.vars_s[bidder_idx, layer_idx, out_feat],
                        f"Affine_{bidder_idx}_{layer_idx}_{out_feat}",
                    )
                    # indicator constraints
                    lhs = self.vars_z[bidder_idx, layer_idx, out_feat]
                    rhs = (
                        self.vars_y[bidder_idx, layer_idx, out_feat]
                        * self.upper_bounds_z[bidder_idx, layer_idx, out_feat]
                    )
                    self.mip.addConstr(
                        lhs <= rhs, f"z_{bidder_idx}_{layer_idx}_{out_feat}_binary"
                    )

                    lhs = self.vars_s[bidder_idx, layer_idx, out_feat]
                    rhs = (
                        1 - self.vars_y[bidder_idx, layer_idx, out_feat]
                    ) * self.upper_bounds_s[bidder_idx, layer_idx, out_feat]
                    self.mip.addConstr(
                        lhs <= rhs, f"s_{bidder_idx}_{layer_idx}_{out_feat}_binary"
                    )
            layer_idx += 1


class WDPGreedyHeuristic:
    """
    Solve the WDP using the following greedy heuristic:
    1. consider all single-item bundles and retrieve valuations using the valuation_models
    2. assign the bundle-bidder combination with the highest valuation
    3. with all remaining items, repeat the process, always considering the already assigned bundles when retrieving
    valuations from the models
    """

    def __init__(
        self, valuation_models: list, free_disposal=False, verbose=0, **kwargs
    ):
        self.name = self.__class__.__name__
        self.valuation_models = valuation_models
        self.free_disposal = free_disposal
        self.verbose = verbose
        self.num_items = self.valuation_models[0].input_dim
        self.num_bidders = len(self.valuation_models)
        self.solutions = []
        self.item_allocations = []
        self.objVals = []
        self.runtime = None
        self._solve()
        pass

    def _solve(self):
        single_item_bundles = {
            j: torch.tensor(
                [0 if i != j else 1 for i in range(self.num_items)],
                device=device,
                dtype=torch.float32,
            )
            for j in range(self.num_items)
        }
        item_allocation = torch.zeros(
            (self.num_bidders, self.num_items), device=device, dtype=torch.float32
        )
        start_time = time.time()
        while single_item_bundles:
            best_valuation = -np.inf
            best_bidder = None
            best_item = None
            best_bundle = None
            for bidder_idx in range(self.num_bidders):
                without_bundle = self.valuation_models[bidder_idx].predict(
                    item_allocation[bidder_idx]
                )
                for item_idx, bundle in single_item_bundles.items():
                    with_bundle = self.valuation_models[bidder_idx].predict(
                        item_allocation[bidder_idx] + bundle
                    )
                    valuation = with_bundle - without_bundle
                    if valuation > best_valuation:
                        best_valuation = valuation
                        best_bidder = bidder_idx
                        best_item = item_idx
                        best_bundle = bundle
            if best_bundle is None:
                break
            item_allocation[best_bidder] += best_bundle
            del single_item_bundles[best_item]
        self.runtime = time.time() - start_time
        self.item_allocations.append(
            [tuple(x) for x in item_allocation.type(torch.int).tolist()]
        )
        objVal = 0
        for bidder_idx in range(self.num_bidders):
            objVal += self.valuation_models[bidder_idx].predict(
                item_allocation[bidder_idx]
            )
        self.objVals.append(objVal.item())
        pass
