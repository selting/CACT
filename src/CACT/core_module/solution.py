import datetime as dt
import json
from typing import Sequence

import pyvrp
from matplotlib import pyplot as plt

import utility_module.io as io
from auction_module.bundle_generation.assignment_based.assignment import Assignment
from auction_module.request_selection.request_selection import RequestSelectionStrategy
from core_module.carrier import Carrier
from core_module.instance import CAHDInstance
from core_module.request import Request
from core_module.tour import Tour
from routing_module.routing_solver import RoutingSolver


class CAHDSolution:
    def __init__(self, instance: CAHDInstance,
                 routing_solver: RoutingSolver,
                 request_selection_strategy: RequestSelectionStrategy = None,
                 objective: str = 'duration', ):
        self.id_: str = instance.id_
        self.meta: dict[str, int] = instance.meta
        assert objective in ['duration', 'distance'], \
            f'Objective must be either "duration" or "distance", not {objective}'
        self._objective = objective

        self._carriers = []
        for idx in range(instance.num_carriers):
            carrier = Carrier(idx, str(idx), instance.depots[idx], objective, instance.carriers_max_num_tours,
                              routing_solver, request_selection_strategy)
            self._carriers.append(carrier)

    def __str__(self):
        s = f'Solution {self.id_}\nObjective={self.objective}'
        s += '\n'
        for c in self.carriers:
            s += str(c)
            s += '\n'
        return s

    def __repr__(self):
        return f'CAHDSolution for {self.id_}'

    @property
    def carriers(self):
        return tuple(self._carriers)

    @property
    def objective(self):
        if self._objective == 'duration':
            return self.sum_travel_duration
        elif self._objective == 'distance':
            return self.sum_travel_distance

    @property
    def metrics(self):
        """
        Returns a dictionary of all metrics that are calculated from the solution. These are the values that are
        typically used to evaluate the quality of a solution. The metrics are calculated for the entire solution, not
        for individual carriers.

        Implemented as a property to allow calling the required methods to calculate the metrics. This way, the metrics
        are always up-to-date when accessed.
        :return:
        """
        return {
            # 'objective': self.objective,
            'sum_travel_distance': self.sum_travel_distance,
            'sum_travel_duration': self.sum_travel_duration,
            'sum_wait_duration': self.sum_wait_duration(),
            'sum_service_duration': self.sum_service_duration(),
            'sum_idle_duration': self.sum_idle_duration(),
            # 'sum_revenue': self.sum_revenue(),
            'utilization': self.utilization(),
            'num_tours': self.num_tours(),
            # 'num_routing_stops': self.num_routing_stops(),
            # 'acceptance_rate': self.avg_acceptance_rate(),
        }

    @property
    def request_to_carrier_assignment(self):
        assignment = Assignment(self.carriers)
        for carrier in self.carriers:
            for request in carrier.assigned_requests:
                assignment[request] = carrier
        return assignment

    @property
    def sum_travel_distance(self):
        return sum(c.sum_travel_distance for c in self.carriers)

    @property
    def sum_travel_duration(self):
        return sum((c.sum_travel_duration for c in self.carriers), dt.timedelta(0))

    def sum_wait_duration(self):
        return sum((c.sum_wait_duration() for c in self.carriers), dt.timedelta(0))

    def sum_service_duration(self):
        return sum((c.sum_service_duration() for c in self.carriers), dt.timedelta(0))

    def sum_idle_duration(self):
        return sum((c.sum_idle_duration() for c in self.carriers), dt.timedelta(0))

    def sum_load(self):
        return sum(c.sum_load() for c in self.carriers)

    def sum_revenue(self):
        return sum(c.sum_revenue() for c in self.carriers)

    def utilization(self):
        """
        average utilization of available resources. See CAHDSolution.utilization() for more information on how the
        carrier utilization is defined
        """
        utilization = [c.utilization() for c in self.carriers]
        return sum(utilization) / len(utilization)

    # def objective(self):
    #     return sum(c.objective for c in self.carriers)

    # def sum_profit(self):
    #     return sum(c.sum_profit() for c in self.carriers)

    def num_carriers(self):
        return len(self.carriers)

    def num_tours(self):
        return len(self.tours)

    def num_routing_stops(self):
        return sum(c.num_routing_stops() for c in self.carriers)

    def avg_acceptance_rate(self):
        # average over all carriers
        return sum([c.acceptance_rate for c in self.carriers]) / self.num_carriers()

    def as_dict(self):
        """The solution as a nested python dictionary"""
        return {carrier.id_: carrier.as_dict() for carrier in self.carriers}

    def summary(self):
        summary = {**self.meta, }
        summary.update(self.solver_config)
        summary.update({
            # 'num_carriers': self.num_carriers(),
            'objective': self.objective,
            # 'sum_profit': self.sum_profit(),
            'sum_travel_distance': self.sum_travel_distance,
            'sum_travel_duration': self.sum_travel_duration,
            'sum_wait_duration': self.sum_wait_duration(),
            'sum_service_duration': self.sum_service_duration(),
            'sum_idle_duration': self.sum_idle_duration(),
            'sum_load': self.sum_load(),
            'sum_revenue': self.sum_revenue(),
            'utilization': self.utilization(),
            'num_tours': self.num_tours(),
            'num_routing_stops': self.num_routing_stops(),
            'acceptance_rate': self.avg_acceptance_rate(),
            **self.logger,
            'carrier_summaries': {c.id_: c.summary() for c in self.carriers}
        })

        return summary

    def to_json(self):
        path = io.cr_ahd_solution_dir.joinpath(self.id_ + '_' + self.solver_config['solution_algorithm'])
        path = io.unique_path(path.parent, path.stem + '_#{:03d}' + '.json')
        with open(path, mode='w') as f:
            json.dump({'summary': self.summary(), 'solution': self.as_dict()}, f, indent=4, cls=io.MyJSONEncoder)
        pass

    def clear_carrier_routes(self, carriers: Sequence[Carrier]):
        """
        delete all existing routes of the given carrier and move all accepted requests to the list of unrouted requests
        :param carriers:
        """
        if carriers is None:
            carriers = self.carriers

        for carrier in carriers:
            carrier.clear_routes()

    def drop_empty_tours_and_adjust_ids(self):
        """
        drops all tours that only contain a depot and do not visit any customer location. removes them from
        self as well as all carriers in self.carriers.
        """
        # update the tour ids TODO is that a good idea?
        for index, tour in enumerate(self.tours):
            tour.id_ = index
        for carrier in self.carriers:
            carrier.drop_empty_tours()

    @property
    def tours(self):
        return [tour for carrier in self.carriers for tour in carrier.tours]

    def tour_of_request(self, request: Request) -> Tour:
        for tour in self.tours:
            if request in tour.requests:
                return tour
        return None

    def unrouted_requests(self):
        for c in self.carriers:
            for r in c.unrouted_requests:
                yield r

    @classmethod
    def from_pyvrp(cls, instance: CAHDInstance, solution: pyvrp.Solution, problem_data: pyvrp.ProblemData,
                   objective: str):
        """
        Create a CAHDSolution from a PyVRP solution. This is the only way to create a CAHDSolution currently.
        The PyVRP solution is a scaled solution, i.e. the travel distance and duration are scaled by the number of
        carriers. The solution is scaled back to the original values.

        :param instance: The instance the solution is based on
        :param solution: The PyVRP solution
        :param objective: The objective of the solution, either 'duration' or 'distance'
        :return: CAHDSolution
        """
        carriers_tours: dict[int, list[Tour]] = {i: [] for i in range(len(problem_data.depots()))}

        for pyvrp_route_idx, pyvrp_route in enumerate(solution.routes()):
            depot_idx: int = pyvrp_route.start_depot()
            depot = instance.depots[depot_idx]

            tour = Tour(pyvrp_route_idx, depot)

            for insertion_idx, pyvrp_client_idx in enumerate(pyvrp_route.visits(), start=1):
                pyvrp_location = problem_data.location(pyvrp_client_idx)
                request = instance.vertices[int(pyvrp_location.name)]
                if tour.insertion_feasibility_check(instance, [insertion_idx], [request]):
                    tour.insert_and_update(instance, [insertion_idx], [request])
                else:
                    print(f'Insertion of request {request} at position {insertion_idx} in tour {tour} is not feasible')
                    raise ValueError

            carriers_tours[depot_idx].append(tour)

        output = cls(instance, objective)

        for carrier in output.carriers:
            for tour in carriers_tours[carrier.id_]:
                carrier.tours.append(tour)
                carrier.accepted_requests.extend(tour.requests)
                carrier.routed_requests.extend(tour.requests)
                for request in tour.requests:
                    output.request_to_carrier_assignment[request] = carrier

        return output

    def plot(self, ax: plt.Axes = None, print_stats=('sum_travel_duration', 'sum_travel_distance')):
        """
        Plot the solution using matplotlib. The solution is plotted as a scatter plot of the requests and the tours
        are plotted as lines connecting the requests.
        :param ax: The axes to plot on. If None, a new figure and axes are created.
        :return: The figure and axes of the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        for carrier, color in zip(self.carriers, ['red', 'blue', 'green']):
            carrier.plot(ax=ax, color=color)

        # add the duration and distance as text to the plot
        vert = 1
        for stat in print_stats:
            stat_value = getattr(self, stat)
            if isinstance(stat_value, dt.timedelta):  # format as nice looking string representation
                stat_value = str(stat_value)
            else:
                stat_value = f'{stat_value:.2f}'
            ax.text(0.5, vert, f'{stat}: {stat_value}',
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='bottom', horizontalalignment='center')
            vert += 0.05

        return fig, ax
