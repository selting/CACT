import warnings
from abc import abstractmethod
from copy import deepcopy
from datetime import timedelta
from pprint import pformat

import mlflow
import numpy as np
import pyvrp
from matplotlib import pyplot as plt
from pyvrp.plotting import *
from pyvrp.stop import MaxRuntime

from auction_module.auction import Auction
from auction_module.request_selection.request_selection import RequestSelectionStrategy
from core_module.carrier import Carrier
from core_module.instance import CAHDInstance
from core_module.request import Request
from core_module.solution import CAHDSolution
from routing_module.routing_solver import RoutingSolver
from tw_management_module.tw_selection import TWSelectionBehavior
from utility_module import utils as ut, profiling as pr
from utility_module.parameterized_class import ParameterizedClass, flatten_dict
from utility_module.pyvrp_helper import scale_problem_data_dict
from utility_module.utils import ACCEPTANCE_START_TIME, generate_time_windows


class Solver(ParameterizedClass):
    """Abstract class for all solvers"""

    def __init__(
        self,
        routing_solver: RoutingSolver,
        time_window_selection: TWSelectionBehavior,
        time_window_length,
    ):
        self.routing_solver: RoutingSolver = routing_solver
        self.time_window_selection: TWSelectionBehavior = time_window_selection
        self.time_window_length: timedelta = time_window_length
        self.auction = None

        self._params = {
            "time_window_selection": time_window_selection,
            "time_window_length": time_window_length,
            "routing_solver": routing_solver,
        }

    def execute(self, instance: CAHDInstance):
        mlflow.log_params(flatten_dict(self.params, "solver", "__"))
        self._execute(instance)
        pass

    @abstractmethod
    def _execute(self, instance: CAHDInstance):
        pass


class IsolatedSolver(Solver):
    """Solving the CR AHD problem in an isolated manner, i.e. without any interaction between carriers"""

    def _execute(self, instance: CAHDInstance):
        """
        apply the concrete steps of the solution algorithms specified in the config

        :return a triple of the adjusted instance with its newly assigned time windows, the solution, and the auction
        """

        # ===== [0] Setup =====
        solution = CAHDSolution(instance, self.routing_solver)

        # ===== Dynamic Request Arrival Phase =====
        self.request_arrival(instance, solution)

        # ===== Final Improvement =====
        objective = solution.objective
        # TODO
        # timer = pr.Timer()
        # for carrier in solution.carriers:
        #     carrier.tour_improvement(instance, solution)
        # timer.stop()
        # mlflow.log_metric('runtime_final_improvement', timer.duration)

        assert objective >= solution.objective, (
            f"{objective} < {solution.objective} but should be >=!({instance.id_, self.params})"
        )
        ut.validate_solution(instance, solution)  # safety check

        print("SUCCESS")
        pass

    def request_arrival(self, instance: CAHDInstance, solution: CAHDSolution):
        tw_options = generate_time_windows(self.time_window_length)

        for request in sorted(instance.requests, key=lambda x: x.disclosure_time):
            carrier: Carrier = solution.carriers[request.initial_carrier_assignment]

            tw_offers = carrier.offer_time_windows(instance, request, tw_options)
            if tw_offers:
                selected_tw = self.time_window_selection.select_tw(tw_offers, request)
                if selected_tw:
                    request.tw_open = selected_tw.open
                    request.tw_close = selected_tw.close
                    carrier.assign_request(request)
                    carrier.accept_request(request)
                    carrier.route_new_request(instance, request)
            else:
                # if no tws are offered, or no tw was selected by the customer, the request is not accepted, tw is reset
                request.tw_open = ut.EXECUTION_TIME_HORIZON.open
                request.tw_close = ut.EXECUTION_TIME_HORIZON.close
                # TODO: below is outdated. check whether the request is in any way connected to the carrier. if not,
                #  simply resetting the tw is enough
                warnings.warn(
                    "Deprecated code! - does not adhere to the latest updates of the carrier class and the "
                    "routing solver. Check and update the code here"
                )
                carrier.reject_request(request)
                pass
        pass


class CollaborativeSolver(IsolatedSolver):
    """Solving the CR AHD problem in a collaborative manner, i.e. with interaction between carriers"""

    def __init__(
        self,
        routing_solver: RoutingSolver,
        time_window_selection: TWSelectionBehavior,
        time_window_length: timedelta,
        request_selection_strategy: RequestSelectionStrategy,
        auction: Auction = False,
    ):
        super().__init__(routing_solver, time_window_selection, time_window_length)
        self._request_selection_strategy = request_selection_strategy
        self.auction: Auction = auction

    def _execute(self, instance: CAHDInstance):
        """
        apply the concrete steps of the solution algorithms specified in the config

        return a triple of the adjusted instance with its newly assigned time windows, the solution, and the auction
        """
        # ===== [0] Setup =====
        solution = CAHDSolution(
            instance, self.routing_solver, self._request_selection_strategy
        )

        # ===== Dynamic Request Arrival Phase =====
        self.request_arrival(instance, solution)

        # ===== Final Improvement =====
        # TODO
        # for carrier in solution.carriers:
        #     carrier.tour_improvement(instance, solution)

        # ===== Final Auction =====
        if self.auction:
            timer = pr.Timer()
            solution = self.auction.run_auction(instance, solution)
            timer.stop()
            mlflow.log_metric("runtime_auction", timer.duration)

        ut.validate_solution(
            instance, solution
        )  # safety check to make sure everything's functional
        pass


class CentralSolverPyVrp(IsolatedSolver):
    """
    TODO: this class needs an overhaul
    Solver class that acts as a central entity having access to all requests and all depots.
    Vehicles cannot be moved between depots, i.e. every depot still is limited to carrier_max_num_vehicles tours.
    Uses PyVRP to solve the routing problem.
    """

    def execute(
        self, instance: CAHDInstance
    ) -> tuple[CAHDInstance, CAHDSolution, None]:
        """
        apply the concrete steps of the solution algorithms specified in the config

        :return a tuple of the adjusted instance with its newly assigned time windows and the solution
        """
        raise DeprecationWarning("This class is deprecated and should not be used")

        # ===== Setup =====
        instance = deepcopy(instance)
        aux_solution = CAHDSolution(
            instance, self.routing_solver, self._request_selection_strategy
        )

        print("SOLVING")

        # ===== Dynamic Request Arrival Phase =====
        for request in sorted(instance.requests, key=lambda x: x.disclosure_time):
            assert request in aux_solution.unassigned_requests

            self.request_acceptance_and_TW(instance, aux_solution, request)

        # ===== Central Routing =====
        timer = pr.Timer()
        #
        pyvrp_problem_data_dict = self.generate_pyvrp_problem_data_dict(
            instance, aux_solution
        )
        pyvrp_problem_data = pyvrp.ProblemData(
            clients=[pyvrp.Client(**c) for c in pyvrp_problem_data_dict["clients"]],
            depots=[pyvrp.Depot(**d) for d in pyvrp_problem_data_dict["depots"]],
            vehicle_types=[
                pyvrp.VehicleType(**v) for v in pyvrp_problem_data_dict["vehicle_types"]
            ],
            distance_matrices=pyvrp_problem_data_dict["distance_matrices"],
            duration_matrices=pyvrp_problem_data_dict["duration_matrices"],
        )

        round_func = "exact"
        pyvrp_problem_data_dict_scaled = scale_problem_data_dict(
            pyvrp_problem_data_dict, round_func=round_func
        )
        pyvrp_problem_data_scaled = pyvrp.ProblemData(
            clients=[
                pyvrp.Client(**c) for c in pyvrp_problem_data_dict_scaled["clients"]
            ],
            depots=[pyvrp.Depot(**d) for d in pyvrp_problem_data_dict_scaled["depots"]],
            vehicle_types=[
                pyvrp.VehicleType(**v)
                for v in pyvrp_problem_data_dict_scaled["vehicle_types"]
            ],
            distance_matrices=pyvrp_problem_data_dict_scaled["distance_matrices"],
            duration_matrices=pyvrp_problem_data_dict_scaled["duration_matrices"],
        )
        pyvrp_model = pyvrp.Model.from_data(pyvrp_problem_data_scaled)
        pyvrp_result = pyvrp_model.solve(
            stop=MaxRuntime(self._params["max_runtime"]), display=True
        )
        pyvrp_solution = pyvrp_result.best

        assert pyvrp_solution.is_feasible(), (
            f"PyVRP solution is infeasible! {pyvrp_solution}"
        )
        plot_solution(pyvrp_solution, pyvrp_problem_data_scaled, False)
        plt.show()
        output_solution = CAHDSolution.from_pyvrp(
            instance,
            pyvrp_solution,
            pyvrp_problem_data,
            objective=aux_solution._objective,
        )

        # timer.write_duration_to_solution(output_solution, 'runtime_final_improvement')
        assert aux_solution.objective >= output_solution.objective, (
            f"{aux_solution.objective} < {output_solution.objective} but should be >=!({instance.id_, self.params})"
        )
        ut.validate_solution(
            instance, output_solution
        )  # safety check to make sure everything's functional

        logger.log(
            SUCCESS,
            f"{instance.id_}: Success\n{pformat(output_solution.solver_config, sort_dicts=True)}",
        )

        pass

    def generate_pyvrp_problem_data_dict(self, instance, pre_solve_solution):
        """
        returns a dictionary that holds all data to create a PyVrp ProblemData instance

        :param instance:
        :param pre_solve_solution:
        :return:
        """
        # depots
        pyvrp_depots_dicts = [dict(x=d.x, y=d.y, name=str(d)) for d in instance.depots]

        # clients
        pyvrp_clients_dicts = []
        sorted_requests = []  # need the Requests to in the same order as the clients
        for carrier in pre_solve_solution.carriers:
            for request in carrier.accepted_requests:
                client = dict(
                    x=request.x,
                    y=request.y,
                    delivery=request.load,
                    service_duration=request.service_duration.total_seconds(),
                    tw_early=(request.tw_open - ACCEPTANCE_START_TIME).total_seconds(),
                    tw_late=(request.tw_close - ACCEPTANCE_START_TIME).total_seconds(),
                    prize=request.revenue,
                    name=str(
                        request.uid
                    ),  # NOTE vertex number is the request id, not the request number
                )
                pyvrp_clients_dicts.append(client)
                sorted_requests.append(request)

        # edges between locations
        distance_matrix = []
        duration_matrix = []
        for i_pyvrp_location, i_vertex in zip(
            pyvrp_depots_dicts + pyvrp_clients_dicts, instance.depots + sorted_requests
        ):
            distance_matrix_row = []
            duration_matrix_row = []
            for j_pyvrp_location, j_vertex in zip(
                pyvrp_depots_dicts + pyvrp_clients_dicts,
                instance.depots + sorted_requests,
            ):
                if i_pyvrp_location == j_pyvrp_location:
                    distance_ij = 0
                    duration_ij = 0
                else:
                    distance_ij = instance.travel_distance([i_vertex], [j_vertex])
                    duration_ij = instance.travel_duration(
                        [i_vertex], [j_vertex]
                    ).total_seconds()
                distance_matrix_row.append(distance_ij)
                duration_matrix_row.append(duration_ij)
            distance_matrix.append(distance_matrix_row)
            duration_matrix.append(duration_matrix_row)

        distance_matrix = np.array(distance_matrix)
        duration_matrix = np.array(duration_matrix)

        # vehicle types
        pyvrp_vehicle_types_dicts = []
        for depot_idx, (carrier, depot) in enumerate(
            zip(pre_solve_solution.carriers, pyvrp_depots_dicts)
        ):
            vehicle_type = dict(
                num_available=instance.carriers_max_num_tours,
                capacity=instance.max_vehicle_load,
                start_depot=depot_idx,
                end_depot=depot_idx,
                tw_early=(
                    carrier.depot.tw_open - ACCEPTANCE_START_TIME
                ).total_seconds(),
                tw_late=(
                    carrier.depot.tw_close - ACCEPTANCE_START_TIME
                ).total_seconds(),
                max_distance=instance.max_tour_distance,
                max_duration=instance.max_tour_duration.total_seconds(),
                unit_distance_cost=0,
                unit_duration_cost=1,
                name=str(carrier.id_),
            )
            pyvrp_vehicle_types_dicts.append(vehicle_type)

        problem_data_dict = dict(
            clients=pyvrp_clients_dicts,
            depots=pyvrp_depots_dicts,
            vehicle_types=pyvrp_vehicle_types_dicts,
            distance_matrices=[distance_matrix],
            duration_matrices=[duration_matrix],
        )

        return problem_data_dict

    def request_acceptance_and_TW(
        self, instance: CAHDInstance, solution: CAHDSolution, request: Request
    ):
        acceptance_type, selected_tw = self.request_acceptance.execute_central(
            instance, solution, request
        )

        if acceptance_type == "accept_feasible":
            request.tw_open = selected_tw.open
            request.tw_close = selected_tw.close
            carrier, tour, pos, delta = (
                self.routing_solver.best_insertion_for_request_in_central_solution(
                    instance, solution, request
                )
            )
            solution.assign_requests_to_carriers({request: carrier}, as_accepted=True)
            if tour is None:
                self.routing_solver.create_new_tour_with_request(
                    instance, solution, carrier, request
                )
            else:
                self.routing_solver.execute_insertion(
                    instance, carrier, request, tour, pos
                )

        else:
            request.tw_open = ut.EXECUTION_TIME_HORIZON.open
            request.tw_close = ut.EXECUTION_TIME_HORIZON.close
