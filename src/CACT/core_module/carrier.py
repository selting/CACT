import datetime as dt

import pandas as pd
from matplotlib import pyplot as plt

from auction_module.bundle_generation.bundle_based.bundle import Bundle
from auction_module.request_selection.request_selection import RequestSelectionStrategy
from routing_module.routing_solver import RoutingSolver
from tw_management_module.time_window import TimeWindow
from .depot import Depot
from .instance import CAHDInstance
from .request import Request
from .tour import Tour


class Carrier:
    """Referred to as 'carrier' throughout the code. Represents a single carrier in the solution."""

    def __init__(
        self,
        carrier_index: int,
        label: str,
        depot: Depot,
        objective: str,
        max_num_tours: int,
        routing_solver: RoutingSolver,
        request_selection_strategy: RequestSelectionStrategy = None,
    ):
        self.id_ = carrier_index
        self.label = label
        self.depot = depot
        # TODO: having assert statements in "production" code is bad style! Check python args -O -OO, and replace MOST
        #  asserts with exceptions
        assert objective in ["distance", "duration"], (
            f'Objective must be either "distance" or "duration", not {objective}'
        )
        self._objective = objective

        self.max_num_tours = max_num_tours
        self._routing_solver = routing_solver
        self._request_selection_strategy = request_selection_strategy

        self._assigned_requests: list[Request] = []
        self._accepted_requests: list[Request] = []
        self._rejected_requests: list[Request] = []
        self._unrouted_requests: list[Request] = []
        self._routed_requests: list[Request] = []
        self._tours: list[Tour] = []

    def __str__(self):
        s = f"Carrier {self.id_}"
        return s

    def __repr__(self):
        return f"Carrier {self.id_}"

    @property
    def assigned_requests(self):
        return tuple(self._assigned_requests)

    @property
    def accepted_requests(self):
        return tuple(self._accepted_requests)

    @property
    def rejected_requests(self):
        return tuple(self._rejected_requests)

    @property
    def unrouted_requests(self):
        # TODO replace the ._unrouted_requests class member and retrieve it dynamically in this property function.
        #  -> no need to always manually update the _unrouted_requests list. unrouted = accepted but not yet routed
        return tuple(self._unrouted_requests)

    @property
    def routed_requests(self):
        # TODO retrieve routed requests dynamically: r for r in t.requests for t in self tours
        return tuple(self._routed_requests)

    @property
    def acceptance_rate(self):
        return len(self.accepted_requests) / len(self.assigned_requests)

    @property
    def objective(self):
        if self._objective == "distance":
            return self.sum_travel_distance
        elif self._objective == "duration":
            return self.sum_travel_duration

    @property
    def tours(self):
        return tuple(self._tours)

    def num_routing_stops(self):
        regular = sum(t.num_routing_stops for t in self._tours)
        return regular

    @property
    def sum_travel_distance(self):
        regular = sum(t.sum_travel_distance for t in self._tours)
        return regular

    @property
    def sum_travel_duration(self):
        regular = sum((t.sum_travel_duration for t in self._tours), dt.timedelta(0))
        return regular

    def sum_wait_duration(self):
        regular = sum((t.sum_wait_duration for t in self._tours), dt.timedelta(0))
        return regular

    def sum_service_duration(self):
        regular = sum((t.sum_service_duration for t in self._tours), dt.timedelta(0))
        return regular

    def sum_idle_duration(self):
        regular = sum((t.sum_idle_duration for t in self._tours), dt.timedelta(0))
        return regular

    def utilization(self):
        """
        Utilization of available capacity. Tour utilization is measured in % of active (i.e., travel + service) time.
        Utilization of the carrier is measures as the average utilization of all possibly available max_num_tours.
        """
        return sum([t.utilization for t in self._tours]) / self.max_num_tours

    def sum_load(self):
        regular = sum(t.sum_load for t in self._tours)
        return regular

    def sum_revenue(self):
        regular = sum(t.sum_revenue for t in self._tours)
        return regular

    def as_dict(self):
        return {
            "id_": self.id_,
            "tours": {tour.id_: tour.as_dict() for tour in self.tours},
        }

    def summary(self) -> dict:
        return {
            "carrier_id": self.id_,
            "num_tours": len(self._tours),
            "num_routing_stops": self.num_routing_stops(),
            # 'sum_profit': self.sum_profit(),
            "sum_travel_distance": self.sum_travel_distance,
            "sum_travel_duration": self.sum_travel_duration,
            "sum_wait_duration": self.sum_wait_duration(),
            "sum_service_duration": self.sum_service_duration(),
            "sum_idle_duration": self.sum_idle_duration(),
            "sum_load": self.sum_load(),
            "sum_revenue": self.sum_revenue(),
            "utilization": self.utilization(),
            "acceptance_rate": self.acceptance_rate,
            "tour_summaries": {t.id_: t.summary() for t in self.tours},
        }

    def add_tour(self, tour: Tour):
        assert tour.routing_sequence[0] == self.depot, (
            f"Tour {tour} does not start at the carrier's depot."
        )
        assert tour.routing_sequence[-1] == self.depot, (
            f"Tour {tour} does not end at the carrier's depot."
        )
        for request in tour.requests:
            assert request in self.unrouted_requests, (
                f"Request {request} is not in the carrier's unrouted requests."
            )
        for request in tour.requests:
            self._unrouted_requests.remove(request)
            self._routed_requests.append(request)
        self._tours.append(tour)

    def drop_empty_tours(self):
        """
        drops all tours that only contain a depot and do not visit any customer location. removes them from
        self
        """
        self._tours = [tour for tour in self.tours if tour.requests]
        pass

    def clear_routes(self):
        """
        resets all routes of the carrier and moves all accepted requests to the list of unrouted requests

        """
        self._unrouted_requests = self._accepted_requests[:]
        self._routed_requests.clear()
        self._tours.clear()

    def assign_request(
        self,
        request: Request,
    ):
        self._assigned_requests.append(request)

    def accept_request(self, request: Request):
        assert request in self.assigned_requests, (
            f"Request {request} is not assigned to carrier {self.id_}."
        )
        self._accepted_requests.append(request)
        self._unrouted_requests.append(request)

    def plot(self, ax: plt.Axes = None, color="tab:red"):
        if ax is None:
            fig, ax = plt.subplots()
        kwargs = dict(c=color, marker="*", zorder=3, s=50)
        ax.scatter(
            self.depot.x,
            self.depot.y,
            label=f"Carrier {self.id_} depot",
            **kwargs,
        )
        for tour in self.tours:
            tour.plot(ax, True, False, False, color)
        if self.unrouted_requests:
            unrouted_xy = [(r.x, r.y) for r in self.unrouted_requests]
            ax.scatter(
                *zip(*unrouted_xy),
                s=40,
                edgecolors=color,
                facecolors="none",
                alpha=0.5,
                label="unrouted",
            )
        # ax.grid(color="grey", linestyle="solid", linewidth=0.2)
        # ax.set_aspect("equal", "datalim")
        # ax.legend(frameon=True)
        # ax.set_title(f'Carrier {self.id_}')
        return ax

    def plot_with_stats(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [4, 1]})
        kwargs = dict(c="tab:red", marker="*", zorder=3, s=200)
        ax1.scatter(
            self.depot.x,
            self.depot.y,
            label=f"Carrier {self.id_} depot",
            **kwargs,
        )
        stats = []
        for tour in self.tours:
            tour.plot(ax1, True, False, False)
            stats.append(
                {
                    "id": tour.id_,
                    "num_stops (w/ depot)": tour.num_routing_stops,
                    "dist": tour.sum_travel_distance,
                    "dur": tour.sum_travel_duration,
                    # 'wait': tour.sum_wait_duration,
                    # 'service': tour.sum_service_duration,
                    # 'idle': tour.sum_idle_duration,
                }
            )

        if self.unrouted_requests:
            unrouted_xy = [(r.x, r.y) for r in self.unrouted_requests]
            ax1.scatter(
                *zip(*unrouted_xy),
                s=40,
                edgecolors="grey",
                facecolors="none",
                alpha=0.5,
                label="unrouted",
            )

        # add the stats as a table below the plot
        if stats:
            df = pd.DataFrame(stats)
            df.set_index("id", inplace=True)
            # add a final row with the sum of all tours
            df.loc["sum"] = df.sum()
            table = ax2.table(
                cellText=df.values,
                colLabels=df.columns,
                rowLabels=df.index,
            )
            # table.auto_set_font_size(False)
            # table.set_fontsize(12)
            # table.scale(1.2, 1.2)
        ax2.axis("off")
        plt.tight_layout()

        ax1.grid(color="grey", linestyle="solid", linewidth=0.2)
        ax1.set_aspect("equal", "datalim")
        ax1.legend(frameon=True)
        ax1.set_title(f"Carrier {self.id_}")
        return fig

    def route_new_request(self, instance: CAHDInstance, request: Request):
        """
        :param instance:
        :return:
        """
        assert request in self.assigned_requests, (
            f"Request {request} is not assigned to carrier {self.id_}."
        )
        assert request in self.accepted_requests, (
            f"Request {request} is not accepted by carrier {self.id_}."
        )
        new_tours = self._routing_solver.insert_request(
            instance, self.tours, self.depot, request, self.max_num_tours
        )
        assert new_tours != "infeasible", f"Inserting request {request} was infeasible."
        self._tours = new_tours
        self._unrouted_requests.remove(request)
        self._routed_requests.append(request)

    def release_requests(self, instance: CAHDInstance, requests: list[Request]):
        """
        releases a request from the carrier. This means that the request is no longer assigned to the carrier, accepted
         by the carrier, and is no longer routed by the carrier. The tour it was removed from simply connects the
        predecessor and successor of the request. The time window of the request is not changed.

        :param requests:
        :return:
        """
        for request in requests:
            assert request in self.assigned_requests, (
                f"Request {request} is not assigned to carrier {self.id_}."
            )
            assert request in self.accepted_requests, (
                f"Request {request} is not accepted by carrier {self.id_}."
            )

        for request in requests:
            if request in self._unrouted_requests:
                self._unrouted_requests.remove(request)
            else:
                # TODO: the logic on how the release is handled in the tours should be defined in the routing_solver!
                tour_of_request: Tour = next(
                    (t for t in self.tours if request in t.requests), None
                )
                tour_of_request.pop_and_update(
                    instance, [tour_of_request.vertex_pos[request]]
                )
                self._routed_requests.remove(request)
            self._accepted_requests.remove(request)
            self._assigned_requests.remove(request)
        pass

    def compute_bid_on_bundle(self, instance: CAHDInstance, bundle: Bundle):
        """
        Computes the bid for a bundle of requests. Assumes truthful bidding.
        :param instance:
        :param bundle:
        :return:
        """
        without_bundle = self.tours
        RS = self._routing_solver
        try:
            if RS.check_bundle_insertion_feasibility(
                instance, without_bundle, self.depot, bundle, self.max_num_tours
            ):
                objective_without_bundle = self.objective
                with_bundle = RS.insert_bundle(
                    instance, without_bundle, self.depot, bundle, self.max_num_tours
                )
                self._tours = list(
                    with_bundle
                )  # temporarily set the tours with the bundle to evaluate their cost
                objective_with_bundle = self.objective
                self._tours = list(without_bundle)

                bid = (objective_with_bundle - objective_without_bundle).total_seconds()
            else:
                bid = "infeasible"
        except ValueError:
            bid = "infeasible"

        return bid

    def route_all_accepted_statically(self, instance: CAHDInstance):
        """
        Routes all accepted requests from scratch using the carrier's static routing strategy

        :param instance:
        :return:
        """
        self.clear_routes()
        new_tours = self._routing_solver.solve_vrp_statically(
            instance, self.depot, self.accepted_requests, self.max_num_tours
        )
        self._tours = new_tours
        for tour in new_tours:
            for request in tour.requests:
                self._unrouted_requests.remove(request)
                self._routed_requests.append(request)
        pass

    def is_time_window_feasible_for_request(
        self, instance: CAHDInstance, request: Request, time_window: TimeWindow
    ):
        """
        previously evaluate_tw_for_carrier() of tw_offering module:
        1) check whether opening a new tour is possible and feasible
        2) check whether inserting the request into an existing tour is possible and feasible
        if any of these is possible, return 1, else -1


        :param request:
        :param time_window:
        :return:
        """
        old_tw_open = request.tw_open
        old_tw_close = request.tw_close
        # temporarily set the time window under consideration
        request.tw_open = time_window.open
        request.tw_close = time_window.close

        feasible = self._routing_solver.check_request_insertion_feasibility(
            instance, self.tours, self.depot, request, self.max_num_tours
        )
        request.tw_open = old_tw_open
        request.tw_close = old_tw_close
        return feasible

    def offer_time_windows(
        self, instance: CAHDInstance, request: Request, tw_options: list[TimeWindow]
    ):
        """
        for each time window in tw_options, check whether the request can be inserted into the carrier's tours.
        return a list of time windows that are feasible for the request.

        :param request:
        :param tw_options:
        :return:
        """
        # this may be extended in the future to include strategic time window management to maximize, e.g. revenue or
        # the number of accepted requests
        tw_offers = []
        for tw in tw_options:
            if self.is_time_window_feasible_for_request(instance, request, tw):
                tw_offers.append(tw)
        return tw_offers

    def select_requests_for_auction(self, instance: CAHDInstance, k: int):
        """
        Select k requests that shall be released to the auction pool
        :param instance:
        :param k:
        :return:
        """
        return self._request_selection_strategy(instance, self.tours, k)

    def route_new_bundle(self, instance: CAHDInstance, bundle: Bundle):
        for request in bundle.requests:
            assert request in self.assigned_requests, (
                f"Request {request} is not assigned to carrier {self.id_}."
            )
            assert request in self.accepted_requests, (
                f"Request {request} is not accepted by carrier {self.id_}."
            )
        new_tours = self._routing_solver.insert_bundle(
            instance, self.tours, self.depot, bundle, self.max_num_tours
        )
        assert new_tours != "infeasible", f"Inserting bundle {bundle} was infeasible."
        self._tours = new_tours
        for request in bundle.requests:
            self._unrouted_requests.remove(request)
            self._routed_requests.append(request)
        pass
