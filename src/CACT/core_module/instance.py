import datetime as dt
import json
import math
from functools import cached_property
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

import utility_module.utils as ut
from core_module.depot import Depot
from core_module.request import Request
from core_module.vertex import Vertex
from utility_module.datetime_handling import ceil_timedelta
from utility_module.io import MyJSONEncoder


class CAHDInstance:
    def __init__(
        self,
        id_: str,
        carriers_max_num_tours: int,
        max_vehicle_load: float,
        max_tour_distance: float,
        max_tour_duration: dt.timedelta,
        requests: list[Request],
        depots: list[Depot],
        duration_matrix,
        distance_matrix,
    ):
        self._id_ = id_
        self.meta = dict(
            (k.strip(), v if k == "t" else int(v.strip()))
            for k, v in (item.split("=") for item in id_.split("+"))
        )
        self.num_carriers = len(depots)
        self.max_vehicle_load = max_vehicle_load
        self.max_tour_distance = max_tour_distance
        self.max_tour_duration = max_tour_duration
        self.carriers_max_num_tours = carriers_max_num_tours

        self.depots: list[Depot] = depots
        self.requests: list[Request] = requests
        self.num_requests = len(self.requests)
        """total number of requests in the instance (across all carriers)"""
        self.vertices = [*self.depots, *self.requests]
        self.num_vertices = len(self.vertices)

        # need to ceil the durations & distances due to floating point precision errors
        self._travel_duration_matrix = np.array(
            [[ceil_timedelta(x, "s") for x in y] for y in duration_matrix]
        )
        assert all(self._travel_duration_matrix.ravel() >= dt.timedelta(0))

        self._travel_distance_matrix = np.array(
            [[math.ceil(x) for x in y] for y in distance_matrix]
        )
        assert all(self._travel_distance_matrix.ravel() >= 0)

        self._validate_data()

    def _validate_data(self):
        # sanity checks:
        for idx, vertex in enumerate(self.depots + self.requests):
            assert vertex.uid == idx, (
                f"Vertex at index {idx} has uid {vertex.uid}. vertices must be sorted by uid "
                f"and uids must be strictly increasing without gaps"
            )
        assert self.requests == sorted(self.requests), (
            f"Request are not sorted by index: {[r.index for r in self.requests]}"
        )
        for idx, request in zip(range(len(self.requests)), self.requests):
            assert request.index == idx, (
                f"Request does not have the correct index. Should be {idx} "
                f"but is {request.index}"
            )
            assert (
                ut.ACCEPTANCE_START_TIME
                <= request.disclosure_time
                <= ut.EXECUTION_START_TIME
            )
            assert ut.EXECUTION_START_TIME <= request.tw_open <= ut.END_TIME
            assert ut.EXECUTION_START_TIME <= request.tw_close <= ut.END_TIME
            assert request.load <= self.max_vehicle_load
        num_vertices = len(self.depots) + len(self.requests)
        assert (
            self._travel_duration_matrix.shape
            == self._travel_distance_matrix.shape
            == (num_vertices, num_vertices)
        )

    pass

    def __repr__(self):
        return f"CAHD Instance {self.id_}({len(self.requests)} customers, {self.num_carriers} carriers)"

    @property
    def id_(self):
        return self._id_

    @cached_property
    def min_x_coord(self):
        return min(r.x for r in self.requests)

    @cached_property
    def max_x_coord(self):
        return max(r.x for r in self.requests)

    @cached_property
    def min_y_coord(self):
        return min(r.y for r in self.requests)

    @cached_property
    def max_y_coord(self):
        return max(r.y for r in self.requests)

    @cached_property
    def min_abs_time_window_distance(self) -> dt.timedelta:
        num_vertices = self.num_requests + self.num_carriers
        return min(
            abs(self.time_window(v0).center - self.time_window(v1).center)
            for v0 in range(num_vertices)
            for v1 in range(num_vertices)
        )

    @cached_property
    def max_abs_time_window_distance(self) -> dt.timedelta:
        num_vertices = self.num_requests + self.num_carriers
        return max(
            abs(self.time_window(v0).center - self.time_window(v1).center)
            for v0 in range(num_vertices)
            for v1 in range(num_vertices)
        )

    @cached_property
    def min_travel_duration(self) -> dt.timedelta:
        return self._travel_duration_matrix.min()

    @cached_property
    def max_travel_duration(self) -> dt.timedelta:
        """maximum travel duration in the instance. This is the maximum travel duration between any two vertices in the
        instance."""
        return self._travel_duration_matrix.max()

    @cached_property
    def nearest_neighbors_durations(self) -> np.ndarray:
        """return for each customer vertex the duration to its nearest neighbor. ambiguous because of asymmetric travel
        duration matrix"""
        requests_travel_duration_matrix = self._travel_duration_matrix[
            self.num_carriers :, self.num_carriers :
        ].copy()
        # ignore distances to self
        np.fill_diagonal(requests_travel_duration_matrix, dt.timedelta.max)
        return requests_travel_duration_matrix.min(axis=1)

    def travel_distance(self, i: Sequence[Vertex], j: Sequence[Vertex]):
        """
        returns the distance between pairs of elements in i and j; i.e., origins are in i and destinations are in j

        Think sum(distance(i[0], j[0]), distance(i[1], j[1]),...)

        """
        distance = 0
        for ii, jj in zip(i, j):
            distance += self._travel_distance_matrix[ii.uid, jj.uid]
        return distance

    def travel_duration(self, i: Sequence[Vertex], j: Sequence[Vertex]):
        """
        returns the distance between pairs of elements in i and j. Think sum(distance(i[0], j[0]), distance(i[1], j[1]),
        ...)

        :param i: Sequence of origin vertex indices
        :param j: Sequence of destination vertex indices

        """
        duration = dt.timedelta(0)
        for ii, jj in zip(i, j):
            duration += self._travel_duration_matrix[ii.uid, jj.uid]
        return duration

    def write_json(self, path: Path):
        data = dict()
        data["_id_"] = self._id_
        data["meta"] = self.meta
        data["num_carriers"] = self.num_carriers
        data["max_vehicle_load"] = self.max_vehicle_load
        data["max_tour_distance"] = self.max_tour_distance
        data["max_tour_duration"] = self.max_tour_duration
        data["carriers_max_num_tours"] = self.carriers_max_num_tours
        data["requests"] = [r.index for r in self.requests]
        data["num_requests"] = self.num_requests
        data["num_requests_per_carrier"] = self.num_requests // self.num_carriers
        data["vertex_x_coords"] = [v.x for v in self.vertices]
        data["vertex_y_coords"] = [v.y for v in self.vertices]
        data["request_to_carrier_assignment"] = [
            r.initial_carrier_assignment for r in self.requests
        ]
        data["request_disclosure_time"] = [
            r.disclosure_time.isoformat() for r in self.requests
        ]
        data["vertex_revenue"] = [0] * self.num_carriers + [
            r.revenue for r in self.requests
        ]
        data["vertex_load"] = [0] * self.num_carriers + [r.load for r in self.requests]
        data["vertex_service_duration"] = [0] * self.num_carriers + [
            r.service_duration.total_seconds() for r in self.requests
        ]
        data["tw_open"] = [v.tw_open.isoformat() for v in self.vertices]
        data["tw_close"] = [v.tw_close.isoformat() for v in self.vertices]
        data["_travel_duration_matrix"] = self._travel_duration_matrix
        data["_travel_distance_matrix"] = self._travel_distance_matrix
        # abort if file already exists
        if path.exists():
            raise FileExistsError(f"File already exists")
        with open(path, "w") as file:
            json.dump(data, file, cls=MyJSONEncoder, indent=4)

    def write_delim(self, path: Path, delim=","):
        """
        it is much easier to read instances from json files

        :param path:
        :param delim:
        :return:
        """
        lines = [
            f"# VRP parameters: V = num of vehicles, L = max_load, T = max_tour_length"
        ]
        lines.extend(
            [
                f"V{delim}{self.carriers_max_num_tours}",
                f"L{delim}{self.max_vehicle_load}",
                f"T{delim}{self.max_tour_distance}\n",
            ]
        )
        lines.extend(
            [
                "# carrier depots: C x y",
                "# one line per carrier, number of carriers defined by number of lines",
            ]
        )
        lines.extend(
            [
                f"C{delim}{x}{delim}{y}"
                for x, y in zip(
                    self.vertex_x_coords[: self.num_carriers],
                    self.vertex_y_coords[: self.num_carriers],
                )
            ]
        )
        lines.extend(
            [
                "\n# requests: carrier_index delivery_x delivery_y revenue",
                "# carrier_index = line index of carriers above",
            ]
        )
        for request in self.requests:
            lines.append(
                f"{self.request_to_carrier_assignment[request]}{delim}"
                f"{self.vertex_x_coords[request + self.num_carriers]}{delim}"
                f"{self.vertex_y_coords[request + self.num_carriers]}{delim}"
                f"{self.vertex_revenue[request + self.num_carriers]}"
            )

        lines.append(
            f"\n# travel duration in seconds. initial entries correspond to depots"
        )

        for i in range(len(self._travel_duration_matrix)):
            lines.append(
                delim.join(
                    [str(x.total_seconds()) for x in self._travel_duration_matrix[i]]
                )
            )

        lines.append(
            f"\n# travel distance in meters. initial entries correspond to depots"
        )

        for i in range(len(self._travel_distance_matrix)):
            # lines.append(delim.join([str(x) for x in self._travel_distance_matrix[i]]))
            lines.append(delim.join(map(str, self._travel_distance_matrix[i])))

        with path.open("w") as f:
            f.writelines([l + "\n" for l in lines])

        pass

    @cached_property
    def _time_window_distance_matrix(self):
        """
        matrix of the pairwise distance between time window centers of all vertices.
        :return:
        """
        matrix = []
        for v0 in range(self.num_requests + self.num_carriers):
            array = []
            for v1 in range(self.num_requests + self.num_carriers):
                array.append(self.time_window(v0).center - self.time_window(v1).center)
            matrix.append(array)
        return np.array(matrix)

    # @cached_property
    # def waiting_time_matrix(self):
    #     pass

    @cached_property
    def LS_vertex_dissimilarity_matrix(self):
        """
        :return: the matrix of pairwise dissimilarity between vertices, where dissimilarity is defined as in:
         Los, Johan, Frederik Schulte, Margaretha Gansterer, Richard F. Hartl, Matthijs T. J. Spaan, and Rudy R.
         Negenborn. 2020. “Decentralized Combinatorial Auctions for Dynamic and Large-Scale Collaborative Vehicle
         Routing.” https://link.springer.com/chapter/10.1007/978-3-030-59747-4_14.
        """

        def LS_waiting_time_seconds(v0: int, v1: int):
            """
            the minimal waiting time (due to time window restrictions) at one of the locations if a vehicle serves
            both locations immediately after each other. Formally, W(i, j) = max(0, min(WD(i, j),WD(j, i)),

            :param v0:
            :param v1:
            :return:
            """
            travel_time = self.travel_duration([v0], [v1])
            if self.tw_open[v0] + travel_time > self.tw_close[v1]:
                return float("inf")
            else:
                t0 = max(
                    self.tw_open[v0] + travel_time, self.tw_open[v1]
                )  # earliest start of service at v2
                t1 = min(
                    self.tw_close[v0] + travel_time, self.tw_close[v1]
                )  # latest start of service at v2
                return (t0 - t1).total_seconds()

        def LS_vertex_dissimilarity(v0: int, v1: int):
            """dissimilarity of two vertices based on a weighted sum of travel time and waiting time. (2 *
            travel_time + minimal_waiting_time)"""

            # w_ij represents the minimal waiting time (due to time window restrictions) at one of the locations,
            # if a vehicle serves both locations immediately after each other
            w_ij = max(
                0, min(LS_waiting_time_seconds(v0, v1), LS_waiting_time_seconds(v1, v0))
            )
            travel_dur = self.travel_duration([v0], [v1])
            gamma = 2  # as in Los et al. (2020)
            return gamma * travel_dur.total_seconds() + w_ij

        num_vertices = self.num_requests + self.num_carriers
        vertex_dissimilarity_matrix = [
            [0.0] * num_vertices for _ in range(num_vertices)
        ]
        for i, v0 in enumerate(range(num_vertices)):
            for j, v1 in enumerate(range(num_vertices)):
                if i == j:
                    continue
                vertex_dissimilarity_matrix[i][j] = LS_vertex_dissimilarity(v0, v1)
        return vertex_dissimilarity_matrix

    @cached_property
    def normalized_travel_duration_matrix(self) -> np.ndarray:
        """min-max normalized travel duration matrix"""
        num_vertices = self.num_requests + self.num_carriers
        matrix = np.zeros((num_vertices, num_vertices), float)
        denom = self.max_travel_duration - self.min_travel_duration
        for v0 in range(num_vertices):
            for v1 in range(num_vertices):
                matrix[v0][v1] = (
                    self._travel_duration_matrix[v0][v1] - self.min_travel_duration
                ) / denom
        return matrix

    @classmethod
    def from_json(cls, path: Path):
        with open(path, "r") as file:
            inst = dict(json.load(file))
        inst["_travel_duration_matrix"] = [
            [dt.timedelta(seconds=y) for y in x]
            for x in inst["_travel_duration_matrix"]
        ]
        inst["max_tour_duration"] = dt.timedelta(seconds=inst["max_tour_duration"])

        depots: list[Depot] = []
        for depot_idx in range(inst["num_carriers"]):
            depots.append(
                Depot(
                    label="Depot " + str(depot_idx),
                    vertex=depot_idx,
                    x=inst["vertex_x_coords"][depot_idx],
                    y=inst["vertex_y_coords"][depot_idx],
                    tw_open=dt.datetime.fromisoformat(inst["tw_open"][depot_idx]),
                    tw_close=dt.datetime.fromisoformat(inst["tw_close"][depot_idx]),
                )
            )

        requests: list[Request] = []
        for request_idx in range(inst["num_requests"]):
            vertex_idx = len(depots) + request_idx
            requests.append(
                Request(
                    vertex_uid=vertex_idx,
                    label=str(request_idx),
                    index=request_idx,
                    x=inst["vertex_x_coords"][vertex_idx],
                    y=inst["vertex_y_coords"][vertex_idx],
                    initial_carrier_assignment=inst["request_to_carrier_assignment"][
                        request_idx
                    ],
                    disclosure_time=dt.datetime.fromisoformat(
                        inst["request_disclosure_time"][request_idx]
                    ),
                    revenue=inst["vertex_revenue"][vertex_idx],
                    load=inst["vertex_load"][vertex_idx],
                    service_duration=dt.timedelta(
                        seconds=inst["vertex_service_duration"][vertex_idx]
                    ),
                    tw_open=dt.datetime.fromisoformat(inst["tw_open"][vertex_idx]),
                    tw_close=dt.datetime.fromisoformat(inst["tw_close"][vertex_idx]),
                )
            )

        return CAHDInstance(
            id_=inst["_id_"],
            carriers_max_num_tours=inst["carriers_max_num_tours"],
            max_vehicle_load=inst["max_vehicle_load"],
            max_tour_distance=inst["max_tour_distance"],
            max_tour_duration=inst["max_tour_duration"],
            requests=requests,
            depots=depots,
            duration_matrix=inst["_travel_duration_matrix"],
            distance_matrix=inst["_travel_distance_matrix"],
        )

    def plot(self):
        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        for carrier_idx in range(self.num_carriers):
            carrier_requests = [
                r for r in self.requests if r.initial_carrier_assignment == carrier_idx
            ]
            xy = [(r.x, r.y) for r in carrier_requests]
            plt.scatter(
                self.depots[carrier_idx].x,
                self.depots[carrier_idx].y,
                marker="s",
                label=f"Depot {self.depots[carrier_idx].label}",
                s=60,
                c=f"C{carrier_idx}",
                edgecolors="k",
                linewidths=1.5,
                zorder=100,
            )
            plt.scatter(
                *zip(*xy), label=f"Carrier {carrier_idx} requests", c=f"C{carrier_idx}"
            )
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

        if self.meta["t"] == "euclidean":
            plt.xlim(0, 25)
            plt.ylim(0, 25)

        # Put a legend to the right of the current axis
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()

        # fig, ax = plt.subplots()
        # for carrier_idx in range(self.num_carriers):
        #     carrier_requests = [r for r in self.requests if r.initial_carrier_assignment == carrier_idx]
        #     xy = [(r.x, r.y) for r in carrier_requests]
        #     plt.scatter(*zip(*xy), label=f'Carrier {carrier_idx} requests', c=f'C{carrier_idx}')
        #     ax.scatter(self.depots[carrier_idx].x, self.depots[carrier_idx].y,
        #                marker='s', label=f'Depot {self.depots[carrier_idx].label}',
        #                s=60, c=f'C{carrier_idx}', edgecolors='k')
        #
        # plt.grid()
        # ax.legend()
        # plt.show()
        pass


if __name__ == "__main__":
    from pprint import pprint

    inst = CAHDInstance.from_json(
        Path(
            "C:/Users/Elting/ucloud/PhD/02_Research/02_Collaborative Routing for Attended Home Deliveries/01_Code/data/CR_AHD_instances/vienna_instances/t=vienna+d=7+c=3+n=10+v=1+o=000+r=00.json"
        )
    )
    print(inst)
    pprint(inst.__dict__)
    pass
