import datetime as dt
from abc import ABC
from copy import deepcopy
from types import MappingProxyType
from typing import Sequence, Set, Union

import pandas as pd
from matplotlib import pyplot as plt

import utility_module.utils as ut
from core_module.depot import Depot
from core_module.instance import CAHDInstance
from core_module.request import Request


class Tour(ABC):
    def __init__(self, id_: int, depot: Depot):
        """

        :param id_: unique tour identifier. (id's can exist twice temporarily if a carrier is copied!)
        :param depot:
        """
        assert isinstance(depot, Depot)

        self.id_ = id_
        self._requests: Set[Request] = set()

        # vertex data
        self._routing_sequence: list[Union[Request, Depot]] = []
        """locations in order of service"""
        self._vertex_pos: dict[Request, int] = dict()
        """mapping each vertex to its routing index"""
        self._arrival_time_sequence: list[dt.datetime] = []
        """arrival time of each vertex"""
        self._service_time_sequence: list[dt.datetime] = []
        """start of service time of each vertex"""
        self._service_duration_sequence: list[dt.timedelta] = []
        """duration of service at each vertex"""
        self._wait_duration_sequence: list[dt.timedelta] = []
        """wait duration at each vertex"""
        self._max_shift_sequence: list[dt.timedelta] = []
        """required for efficient feasibility checks"""

        # sums
        self._sum_travel_distance: float = 0.0
        self._sum_travel_duration: dt.timedelta = dt.timedelta(0)
        self._sum_service_duration: dt.timedelta = dt.timedelta(0)
        self._sum_load: float = 0.0
        self._sum_revenue: float = 0.0
        # self.sum_profit: float = 0.0

        # initialize depot-to-depot tour
        for _ in range(2):
            self._routing_sequence.insert(1, depot)
            self._arrival_time_sequence.insert(1, ut.EXECUTION_START_TIME)
            self._service_time_sequence.insert(1, ut.EXECUTION_START_TIME)
            self._service_duration_sequence.insert(1, dt.timedelta(0))
            self._wait_duration_sequence.insert(1, dt.timedelta(0))
            self._max_shift_sequence.insert(1, ut.END_TIME - ut.EXECUTION_START_TIME)

    def __str__(self):
        return (
            f"Tour ID:\t{self.id_}\n"
            f"Requests:\t{self.requests}\n"
            f"Sequence:\t{self.routing_sequence}\n"
            f"Arrival:\t{[x.strftime('%d-%H:%M:%S') for x in self.arrival_time_sequence]}\n"
            f"Wait:\t\t{[str(x) for x in self.wait_duration_sequence]}\n"
            f"Service Time:\t{[x.strftime('%d-%H:%M:%S') for x in self.service_time_sequence]}\n"
            f"Service Duration:\t{[str(x) for x in self.service_duration_sequence]}\n"
            f"Max Shift:\t{[str(x) for x in self.max_shift_sequence]}\n"
            f"Distance:\t{round(self.sum_travel_distance, 2)}\n"
            f"Duration:\t{self.sum_travel_duration}\n"
            f"Revenue:\t{self.sum_revenue}\n"
        )
        # f'Profit:\t\t{self.sum_profit}\n'

    def __repr__(self):
        return f"Tour {self.id_}: {len(self)} {[v.uid for v in self.routing_sequence]}"

    def __len__(self):
        return len(self.routing_sequence)

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)

        setattr(result, "id_", self.id_)
        setattr(result, "_requests", self._requests.copy())
        setattr(result, "_routing_sequence", self._routing_sequence[:])
        setattr(result, "_vertex_pos", self._vertex_pos.copy())
        setattr(result, "_arrival_time_sequence", self._arrival_time_sequence[:])
        setattr(result, "_service_time_sequence", self._service_time_sequence[:])
        setattr(
            result, "_service_duration_sequence", self._service_duration_sequence[:]
        )
        setattr(result, "_wait_duration_sequence", self._wait_duration_sequence[:])
        setattr(result, "_max_shift_sequence", self._max_shift_sequence[:])
        setattr(result, "_sum_travel_distance", self.sum_travel_distance)
        setattr(result, "_sum_travel_duration", self.sum_travel_duration)
        setattr(result, "_sum_service_duration", self.sum_service_duration)
        setattr(result, "_sum_load", self.sum_load)
        setattr(result, "_sum_revenue", self.sum_revenue)

        return result

    @property
    def requests(self):
        return frozenset(self._requests)

    @property
    def routing_sequence(self):
        return tuple(self._routing_sequence)

    @property
    def vertex_pos(self):
        return MappingProxyType(self._vertex_pos)

    @property
    def arrival_time_sequence(self):
        return tuple(self._arrival_time_sequence)

    @property
    def service_time_sequence(self):
        return tuple(self._service_time_sequence)

    @property
    def service_duration_sequence(self):
        return tuple(self._service_duration_sequence)

    @property
    def wait_duration_sequence(self):
        return tuple(self._wait_duration_sequence)

    @property
    def max_shift_sequence(self):
        return tuple(self._max_shift_sequence)

    @property
    def num_routing_stops(self):
        return len(self)

    @property
    def sum_travel_distance(self):
        return self._sum_travel_distance

    @property
    def sum_travel_duration(self):
        return self._sum_travel_duration

    @property
    def sum_service_duration(self):
        return self._sum_service_duration

    @property
    def sum_load(self):
        return self._sum_load

    @property
    def sum_revenue(self):
        return self._sum_revenue

    @property
    def sum_wait_duration(self):
        return sum(self.wait_duration_sequence, dt.timedelta(0))

    @property
    def sum_idle_duration(self):
        """assumes that vehicles leave the depot immediately! Therefore, there is no idle time at the start depot"""
        idle = ut.END_TIME - (
            self.service_time_sequence[-1] + self.service_duration_sequence[-1]
        )
        return idle

    @property
    def utilization(self):
        # utilization = (self.sum_travel_duration + self.sum_service_duration) / (
        #         self.sum_travel_duration + self.sum_service_duration + self.sum_wait_duration)
        utilization = (
            self.sum_travel_duration + self.sum_service_duration
        ) / ut.EXECUTION_TIME_HORIZON.duration
        return utilization

    def as_dict(self):
        return {
            "routing_sequence": self.routing_sequence,
            "arrival_schedule": self.arrival_time_sequence,
            "wait_sequence": self.wait_duration_sequence,
            "max_shift_sequence": self.max_shift_sequence,
            "service_time_schedule": self.service_time_sequence,
            "service_duration_sequence": self.service_duration_sequence,
        }

    def print_as_table(self):
        def format_timedelta(td):
            """Format timedelta as 'DD-HH:MM:SS'."""
            total_seconds = int(td.total_seconds())
            days, remainder = divmod(total_seconds, 86400)  # 86400 seconds in a day
            hours, remainder = divmod(remainder, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{days:02d}-{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Build a list of dictionaries for table rows
        data = [
            {
                "pos": i,
                "Vertex UID": v.uid,
                "Arrival": a.strftime("%d-%H:%M:%S"),
                "Wait": format_timedelta(w),
                "Service Time": st.strftime("%d-%H:%M:%S"),
                "Service Duration": format_timedelta(sd),
                "Max_Shift": format_timedelta(m),
            }
            for i, (v, a, w, st, sd, m) in enumerate(
                zip(
                    self.routing_sequence,
                    self.arrival_time_sequence,
                    self.wait_duration_sequence,
                    self.service_time_sequence,
                    self.service_duration_sequence,
                    self.max_shift_sequence,
                )
            )
        ]

        # Create a DataFrame from the data
        df = pd.DataFrame(data)

        # Print the table
        print(df.to_string(index=False))

    def summary(self):
        return {
            "tour_id": self.id_,
            # 'sum_profit': self.sum_profit,
            "num_routing_stops": self.num_routing_stops,
            "sum_travel_distance": self.sum_travel_distance,
            "sum_travel_duration": self.sum_travel_duration,
            "sum_wait_duration": self.sum_wait_duration,
            "sum_service_duration": self.sum_service_duration,
            "sum_idle_duration": self.sum_idle_duration,
            "sum_load": self.sum_load,
            "sum_revenue": self.sum_revenue,
            "utilization": self.utilization,
        }

    def _single_insertion_feasibility_check(
        self, instance: CAHDInstance, insertion_index: int, request: Request
    ):
        """
        Checks whether the insertion of the request at insertion_pos is feasible.

        Following
        [1] Vansteenwegen,P., Souffriau,W., Vanden Berghe,G., & van Oudheusden,D. (2009). Iterated local
        search for the team orienteering problem with time windows. Computers & Operations Research, 36(12),
        3281–3290. https://doi.org/10.1016/j.cor.2009.03.008

        [2] Lu,Q., & Dessouky,M.M. (2006). A new insertion-based construction heuristic for solving the pickup and
        delivery problem with time windows. European Journal of Operational Research, 175(2), 672–687.
        https://doi.org/10.1016/j.ejor.2005.05.012

        :return: True if the insertion of the request at insertion_position is feasible, False otherwise
        """

        i: Union[Depot, Request] = self.routing_sequence[insertion_index - 1]
        j: Request = request
        k: Union[Depot, Request] = self.routing_sequence[insertion_index]

        # [1] check max tour distance
        distance_shift_j = instance.travel_distance(
            [i, j], [j, k]
        ) - instance.travel_distance([i], [k])
        if self.sum_travel_distance + distance_shift_j > instance.max_tour_distance:
            return False

        # [2] check max tour travel duration
        j_travel_duration_increase = instance.travel_duration(
            [i, j], [j, k]
        ) - instance.travel_duration([i], [k])
        if (
            self.sum_travel_duration + j_travel_duration_increase
            > instance.max_tour_duration
        ):
            return False

        # [3] check time windows
        # tw condition 1: start of service of j must fit the time window of j
        arrival_time_j = (
            self.service_time_sequence[insertion_index - 1]
            + i.service_duration
            + instance.travel_duration([i], [j])
        )
        tw_cond1 = arrival_time_j <= j.tw_close

        # tw condition 2: time_shift_j must be limited to the sum of wait_k + max_shift_k
        wait_j = max(dt.timedelta(0), j.tw_open - arrival_time_j)
        time_shift_j = (
            instance.travel_duration([i], [j])
            + wait_j
            + j.service_duration
            + instance.travel_duration([j], [k])
            - instance.travel_duration([i], [k])
        )
        wait_k = self.wait_duration_sequence[insertion_index]
        max_shift_k = self.max_shift_sequence[insertion_index]
        tw_cond2 = time_shift_j <= wait_k + max_shift_k

        if not tw_cond1 or not tw_cond2:
            return False

        # [4] check max vehicle load
        if self.sum_load + j.load > instance.max_vehicle_load:
            return False
        return True

    def insertion_feasibility_check(
        self,
        instance: CAHDInstance,
        insertion_indices: Sequence[int],
        requests: Sequence[Request],
    ):
        """
        check whether an insertion of insertion_vertices at insertion_pos is feasible.

        :return: True if the combined insertion of all vertices in their corresponding positions is feasible, False
        otherwise
        """
        if len(insertion_indices) == 1:
            return self._single_insertion_feasibility_check(
                instance, insertion_indices[0], requests[0]
            )
        else:
            # sanity check whether insertion positions are sorted in ascending order
            assert all(
                insertion_indices[i] < insertion_indices[i + 1]
                for i in range(len(insertion_indices) - 1)
            )

            copy = deepcopy(self)

            # check all insertions sequentially
            for idx, (pos, vertex) in enumerate(zip(insertion_indices, requests)):
                if copy._single_insertion_feasibility_check(instance, pos, vertex):
                    if (
                        idx < len(insertion_indices) - 1
                    ):  # to skip the last temporary insertion
                        copy._single_insert_and_update(instance, pos, vertex)
                else:
                    return False
            return True

    def _single_insert_and_update(
        self, instance: CAHDInstance, insertion_index: int, request: Request
    ):
        """
        ASSUMES THAT THE INSERTION WAS FEASIBLE, NO MORE CHECKS ARE EXECUTED IN HERE!

        insert request in a specified position of a routing sequence and update all related sequences, sums and schedules

        Following
        [1] Vansteenwegen,P., Souffriau,W., Vanden Berghe,G., & van Oudheusden,D. (2009). Iterated local search for the
        team orienteering problem with time windows. Computers & Operations Research, 36(12), 3281–3290.
        https://doi.org/10.1016/j.cor.2009.03.008
        [2] Lu,Q., & Dessouky,M.M. (2006). A new insertion-based construction heuristic for solving the pickup and
        delivery problem with time windows. European Journal of Operational Research, 175(2), 672–687.
        https://doi.org/10.1016/j.ejor.2005.05.012

        """

        assert 0 < insertion_index < len(self)
        assert 0 <= request.uid < instance.num_carriers + instance.num_requests * 2, (
            f"Vertex {request.uid} is out of bounds"
        )

        # ===== [1] INSERT =====
        self._requests.add(request)
        self._routing_sequence.insert(insertion_index, request)
        self._vertex_pos[request] = insertion_index

        i_index, i_vertex = (
            insertion_index - 1,
            self._routing_sequence[insertion_index - 1],
        )
        j_index, j_vertex = insertion_index, request
        k_index, k_vertex = (
            insertion_index + 1,
            self._routing_sequence[insertion_index + 1],
        )

        # calculate arrival at j_vertex (cannot use the _service_time_dict because of the depot)
        arrival_j = (
            self.service_time_sequence[i_index]
            + i_vertex.service_duration
            + instance.travel_duration([i_vertex], [j_vertex])
        )
        self._arrival_time_sequence.insert(insertion_index, arrival_j)

        # calculate wait duration at j_vertex
        wait_j = max(dt.timedelta(0), j_vertex.tw_open - arrival_j)
        self._wait_duration_sequence.insert(insertion_index, wait_j)

        # calculate start of service at j_vertex
        service_j = max(j_vertex.tw_open, arrival_j)
        self._service_time_sequence.insert(insertion_index, service_j)

        # store the service duration at j_vertex
        self._service_duration_sequence.insert(
            insertion_index, j_vertex.service_duration
        )

        # set max_shift of j_vertex temporarily to 0, will be updated further down
        max_shift_j = dt.timedelta(0)
        self._max_shift_sequence.insert(insertion_index, max_shift_j)

        # ===== [2] UPDATE =====
        # dist_shift: total distance consumption of inserting j_vertex in between i_vertex and k_vertex
        dist_shift_j = (
            instance.travel_distance([i_vertex], [j_vertex])
            + instance.travel_distance([j_vertex], [k_vertex])
            - instance.travel_distance([i_vertex], [k_vertex])
        )

        # time_shift: total time consumption of inserting j_vertex in between i_vertex and k_vertex
        travel_time_shift_j = (
            instance.travel_duration([i_vertex], [j_vertex])
            + instance.travel_duration([j_vertex], [k_vertex])
            - instance.travel_duration([i_vertex], [k_vertex])
        )
        time_shift_j = travel_time_shift_j + wait_j + j_vertex.service_duration

        # update sums
        self._sum_travel_distance += dist_shift_j
        self._sum_travel_duration += travel_time_shift_j
        self._sum_load += j_vertex.load
        self._sum_revenue += j_vertex.revenue
        self._sum_service_duration += j_vertex.service_duration
        # self.sum_profit = self.sum_profit + instance.vertex_revenue[j_vertex] - time_shift_j

        # update arrival at k_vertex
        arrival_k = self.arrival_time_sequence[k_index] + time_shift_j
        self._arrival_time_sequence[k_index] = arrival_k

        # time_shift_k: how much of j_vertex's time shift is still available after waiting at k_vertex
        time_shift_k = max(
            dt.timedelta(0), time_shift_j - self.wait_duration_sequence[k_index]
        )

        # update waiting time at k_vertex
        wait_k = max(
            dt.timedelta(0), self.wait_duration_sequence[k_index] - time_shift_j
        )
        self._wait_duration_sequence[k_index] = wait_k

        # update start of service at k_vertex
        service_k = self.service_time_sequence[k_index] + time_shift_k
        self._service_time_sequence[k_index] = service_k

        # update max shift of k_vertex
        max_shift_k = self.max_shift_sequence[k_index] - time_shift_k
        self._max_shift_sequence[k_index] = max_shift_k

        # increase vertex position record by 1 for all vertices succeeding j_vertex
        for vertex in self._routing_sequence[insertion_index + 1 : -1]:
            self._vertex_pos[vertex] += 1

        # update data for all visits AFTER j_vertex until (a) shift == 0 or (b) the end is reached
        while time_shift_k > dt.timedelta(0) and k_index + 1 < len(
            self._routing_sequence
        ):
            # move one forward
            k_index += 1
            k_vertex = self._routing_sequence[k_index]
            time_shift_j = time_shift_k

            # update arrival at k_vertex
            arrival_k = self.arrival_time_sequence[k_index] + time_shift_j
            self._arrival_time_sequence[k_index] = arrival_k

            time_shift_k = max(
                dt.timedelta(0), time_shift_j - self.wait_duration_sequence[k_index]
            )

            # update wait duration
            wait_k = max(
                dt.timedelta(0), self.wait_duration_sequence[k_index] - time_shift_j
            )
            self._wait_duration_sequence[k_index] = wait_k

            # update service start time of k_vertex
            service_k = self.service_time_sequence[k_index] + time_shift_k
            self._service_time_sequence[k_index] = service_k

            # update max_shift of k_vertex
            max_shift_k = self.max_shift_sequence[k_index] - time_shift_k
            self._max_shift_sequence[k_index] = max_shift_k

        # update max_shift for visit j_vertex and visits PRECEDING the inserted vertex j_vertex
        for index in range(insertion_index, -1, -1):
            vertex = self._routing_sequence[index]

            max_shift_j = min(
                vertex.tw_close - self.service_time_sequence[index],
                self.wait_duration_sequence[index + 1]
                + self.max_shift_sequence[index + 1],
            )
            self._max_shift_sequence[index] = max_shift_j
        pass

    def insert_and_update(
        self,
        instance: CAHDInstance,
        insertion_indices: Sequence[int],
        requests: Sequence[Request],
    ):
        """
        Inserts insertion_vertices at insertion_indices & updates the necessary data, e.g., arrival times.
        """
        assert all(
            insertion_indices[i] < insertion_indices[i + 1]
            for i in range(len(insertion_indices) - 1)
        )

        # execute all insertions sequentially:
        for index, request in zip(insertion_indices, requests):
            self._single_insert_and_update(instance, index, request)

    def _single_pop_and_update(self, instance: CAHDInstance, pop_index: int):
        """
        Following
        [1] Vansteenwegen,P., Souffriau,W., Vanden Berghe,G., & van Oudheusden,D. (2009). Iterated local search for the team
        orienteering problem with time windows. Computers & Operations Research, 36(12), 3281–3290.
        https://doi.org/10.1016/j.cor.2009.03.008
        [2] Lu,Q., & Dessouky,M.M. (2006). A new insertion-based construction heuristic for solving the pickup and
        delivery problem with time windows. European Journal of Operational Research, 175(2), 672–687.
        https://doi.org/10.1016/j.ejor.2005.05.012

        :return: the popped vertex j at index pop_pos of the routing_sequence, as well as a dictionary of updated sums
        """

        # ===== [1] POP =====
        self._requests.remove(self._routing_sequence[pop_index])
        popped = self._routing_sequence.pop(pop_index)
        self._vertex_pos.pop(popped)

        index = pop_index
        i_vertex = self._routing_sequence[index - 1]
        j_vertex = popped
        k_vertex = self._routing_sequence[
            index
        ]  # k_vertex has taken the place of j_vertex after j_vertex was removed

        self._arrival_time_sequence.pop(pop_index)

        wait_j = self._wait_duration_sequence.pop(pop_index)

        self._service_time_sequence.pop(pop_index)

        self._service_duration_sequence.pop(pop_index)

        self._max_shift_sequence.pop(pop_index)

        # ===== [2] UPDATE =====

        # dist_shift: total distance reduction of removing j_vertex from in between i_vertex and k_vertex
        dist_shift_j = (
            instance.travel_distance([i_vertex], [k_vertex])
            - instance.travel_distance([i_vertex], [j_vertex])
            - instance.travel_distance([j_vertex], [k_vertex])
        )

        # time_shift: total time reduction of removing j_vertex from in between i_vertex and k_vertex
        travel_time_shift_j = (
            instance.travel_duration([i_vertex], [k_vertex])
            - instance.travel_duration([i_vertex], [j_vertex])
            - instance.travel_duration([j_vertex], [k_vertex])
        )

        time_shift_j = travel_time_shift_j - wait_j - j_vertex.service_duration

        # update sums
        self._sum_travel_distance += (
            dist_shift_j  # += since dist_shift_j will be negative
        )
        self._sum_travel_duration += (
            travel_time_shift_j  # += since time_shift_j will be negative
        )
        self._sum_load -= j_vertex.load
        self._sum_revenue -= j_vertex.revenue
        # self.sum_profit = self.sum_profit - instance.vertex_revenue[j_vertex] - dist_shift_j

        # update the arrival at k_vertex
        arrival_k = self.arrival_time_sequence[index] + time_shift_j
        self._arrival_time_sequence[index] = arrival_k

        # update waiting time at k_vertex (more complicated than in insert) - can only increase
        wait_k = max(
            dt.timedelta(0), k_vertex.tw_open - self.arrival_time_sequence[index]
        )
        self._wait_duration_sequence[index] = wait_k

        # time_shift_k: how much of i_vertex's time shift is still available after waiting at k_vertex?
        time_shift_k = min(
            dt.timedelta(0), time_shift_j + self.wait_duration_sequence[index]
        )

        # update start of service at k_vertex
        service_k = max(k_vertex.tw_open, self.arrival_time_sequence[index])
        self._service_time_sequence[index] = service_k

        # update max shift of k_vertex
        max_shift_k = self.max_shift_sequence[index] - time_shift_k
        self._max_shift_sequence[index] = max_shift_k

        # decrease vertex position record by 1 for all vertices succeeding j_vertex
        for vertex in self._routing_sequence[pop_index:-1]:
            self._vertex_pos[vertex] -= 1

        # update data for all visits AFTER j_vertex until (a) shift == 0 or (b) the end is reached
        while time_shift_k < dt.timedelta(0) and index + 1 < len(
            self._routing_sequence
        ):
            # move one forward
            index += 1
            k_vertex = self._routing_sequence[index]
            time_shift_j = time_shift_k

            # update arrival at k_vertex
            arrival_k = self.arrival_time_sequence[index] + time_shift_j
            self._arrival_time_sequence[index] = arrival_k

            # update wait time at k_vertex
            wait_k = max(
                dt.timedelta(0), k_vertex.tw_open - self.arrival_time_sequence[index]
            )
            self._wait_duration_sequence[index] = wait_k

            time_shift_k = min(
                dt.timedelta(0), time_shift_j + self.wait_duration_sequence[index]
            )

            # service start time of k_vertex
            service_k = max(k_vertex.tw_open, self.arrival_time_sequence[index])
            self._service_time_sequence[index] = service_k

            # update max_shift of k_vertex
            max_shift_k = self.max_shift_sequence[index] - time_shift_k
            self._max_shift_sequence[index] = max_shift_k

        # update max_shift for visits PRECEDING the removed vertex j_vertex
        for index in range(pop_index - 1, -1, -1):
            vertex = self._routing_sequence[index]
            max_shift_i = min(
                vertex.tw_close - self.service_time_sequence[index],
                self.wait_duration_sequence[index + 1]
                + self.max_shift_sequence[index + 1],
            )
            self._max_shift_sequence[index] = max_shift_i

        return popped

    def pop_and_update(self, instance: CAHDInstance, pop_indices: Sequence[int]):
        """
        Removes vertices located at pop_indices
        :return: a list of popped vertices
        """

        # assure that indices are sorted
        assert all(
            pop_indices[i] <= pop_indices[i + 1] for i in range(len(pop_indices) - 1)
        )
        popped = []

        # traverse the indices backwards to ensure that the succeeding indices are still correct once preceding ones
        # have been removed
        for pop_index in reversed(pop_indices):
            popped_vertex = self._single_pop_and_update(instance, pop_index)
            popped.append(popped_vertex)

        # reverse the popped array again to return vertices in the expected order
        return list(reversed(popped))

    def pop_distance_delta(self, instance: CAHDInstance, pop_indices: Sequence[int]):
        """
        :return: the negative delta that is obtained by popping the pop_indices from the routing sequence. NOTE: does
        not actually remove/pop the vertices
        """

        delta = 0

        # easy for single insertion
        if len(pop_indices) == 1:
            j_pos = pop_indices[0]
            i_vertex = self._routing_sequence[j_pos - 1]
            j_vertex = self._routing_sequence[j_pos]
            k_vertex = self._routing_sequence[j_pos + 1]

            delta += instance.travel_distance([i_vertex], [k_vertex])
            delta -= instance.travel_distance(
                [i_vertex, j_vertex], [j_vertex, k_vertex]
            )

        # must ensure that no edges are counted twice. Naive implementation with a tmp_routing_sequence.
        else:
            assert all(
                pop_indices[i] < pop_indices[i + 1] for i in range(len(pop_indices) - 1)
            ), f"Pop indices {pop_indices} are not in correct order"

            tmp_routing_sequence = list(self._routing_sequence)

            for j_pos in reversed(pop_indices):
                i_vertex = tmp_routing_sequence[j_pos - 1]
                j_vertex = tmp_routing_sequence.pop(j_pos)
                k_vertex = tmp_routing_sequence[j_pos]

                delta += instance.travel_distance([i_vertex], [k_vertex])
                delta -= instance.travel_distance(
                    [i_vertex, j_vertex], [j_vertex, k_vertex]
                )

        return delta

    def pop_duration_delta(self, instance: CAHDInstance, pop_indices: Sequence[int]):
        """
        :return: the (usually negative) delta that is obtained by popping the pop_indices from the routing sequence.
        """

        delta = dt.timedelta(0)

        # easy for single insertion
        if len(pop_indices) == 1:
            j_pos = pop_indices[0]
            i_vertex = self._routing_sequence[j_pos - 1]
            j_vertex = self._routing_sequence[j_pos]
            k_vertex = self._routing_sequence[j_pos + 1]

            delta += instance.travel_duration([i_vertex], [k_vertex])
            delta -= instance.travel_duration(
                [i_vertex, j_vertex], [j_vertex, k_vertex]
            )

        # must ensure that no edges are counted twice. Naive implementation with a tmp_routing_sequence.
        else:
            assert all(
                pop_indices[i] < pop_indices[i + 1] for i in range(len(pop_indices) - 1)
            ), f"Pop indices {pop_indices} are not in correct order"

            tmp_routing_sequence = list(self._routing_sequence)

            for j_pos in reversed(pop_indices):
                i_vertex = tmp_routing_sequence[j_pos - 1]
                j_vertex = tmp_routing_sequence.pop(j_pos)
                k_vertex = tmp_routing_sequence[j_pos]

                delta += instance.travel_duration([i_vertex], [k_vertex])
                delta -= instance.travel_duration(
                    [i_vertex, j_vertex], [j_vertex, k_vertex]
                )

        return delta

    def insert_duration_delta(
        self,
        instance: CAHDInstance,
        insertion_indices: list[int],
        requests: list[Request],
    ):
        """
        returns the duration surplus that is obtained by inserting the insertion_vertices at the insertion_positions.
        NOTE: Does not perform a feasibility check and does not actually insert the vertices!

        """
        delta = dt.timedelta(0)

        # easy for single insertion
        if len(insertion_indices) == 1:
            j_pos = insertion_indices[0]
            i_vertex = self._routing_sequence[j_pos - 1]
            j_vertex = requests[0]
            k_vertex = self._routing_sequence[j_pos]

            delta += instance.travel_duration(
                [i_vertex, j_vertex], [j_vertex, k_vertex]
            )
            delta -= instance.travel_duration([i_vertex], [k_vertex])

        # must ensure that no edges are counted twice. Naive implementation with a tmp_routing_sequence
        else:
            assert all(
                insertion_indices[i] < insertion_indices[i + 1]
                for i in range(len(insertion_indices) - 1)
            )

            tmp_routing_sequence = list(self._routing_sequence)

            for j_pos, j_vertex in zip(insertion_indices, requests):
                tmp_routing_sequence.insert(j_pos, j_vertex)

                i_vertex = tmp_routing_sequence[j_pos - 1]
                k_vertex = tmp_routing_sequence[j_pos + 1]

                delta += instance.travel_duration(
                    [i_vertex, j_vertex], [j_vertex, k_vertex]
                )
                delta -= instance.travel_duration([i_vertex], [k_vertex])

        return delta

    def insert_distance_delta(
        self,
        instance: CAHDInstance,
        insertion_indices: list[int],
        requests: list[Request],
    ):
        """
        returns the distance surplus that is obtained by inserting the insertion_vertices at the insertion_positions.
        NOTE: Does not perform a feasibility check and does not actually insert the vertices!

        """
        delta = 0

        # trivial for single insertion
        if len(insertion_indices) == 1:
            j_pos = insertion_indices[0]
            i_vertex = self._routing_sequence[j_pos - 1]
            j_vertex = requests[0]
            k_vertex = self._routing_sequence[j_pos]

            delta += instance.travel_distance(
                [i_vertex, j_vertex], [j_vertex, k_vertex]
            )
            delta -= instance.travel_distance([i_vertex], [k_vertex])
            assert delta >= 0

        # must ensure that no edges are counted twice. Naive implementation with a tmp_routing_sequence
        else:
            assert all(
                insertion_indices[i] < insertion_indices[i + 1]
                for i in range(len(insertion_indices) - 1)
            )

            tmp_routing_sequence = list(self._routing_sequence)

            for j_pos, j_vertex in zip(insertion_indices, requests):
                tmp_routing_sequence.insert(j_pos, j_vertex)

                i_vertex = tmp_routing_sequence[j_pos - 1]
                k_vertex = tmp_routing_sequence[j_pos + 1]

                delta += instance.travel_distance(
                    [i_vertex, j_vertex], [j_vertex, k_vertex]
                )
                delta -= instance.travel_distance([i_vertex], [k_vertex])

        return delta

    def _single_insert_max_shift_delta(
        self, instance, insertion_index: int, request: Request
    ):
        """
        returns the change in max_shift time that would be observed if insertion_vertex was placed at
        insertion_index
        """

        # [1] compute wait_j and max_shift_j
        predecessor = self._routing_sequence[insertion_index - 1]
        successor = self._routing_sequence[insertion_index]

        arrival_j = (
            max(self.arrival_time_sequence[insertion_index - 1], predecessor.tw_open)
            + instance.vertex_service_duration[predecessor]
            + instance.travel_duration([predecessor], [request])
        )
        wait_j = max(dt.timedelta(0), request.tw_open - arrival_j)
        delta_ = (
            instance.travel_duration([predecessor], [request])
            + instance.travel_duration([request], [successor])
            - instance.travel_duration([predecessor], [successor])
            + instance.vertex_service_duration[request]
            + wait_j
        )
        max_shift_j = min(
            request.tw_close - max(arrival_j, request.tw_open),
            self.wait_duration_sequence[insertion_index]
            + self.max_shift_sequence[insertion_index]
            - delta_,
        )

        # [2] algorithm 4.2 for max_shift delta of PRECEDING visits:
        # Lu,Q., & Dessouky,M.M. (2006). A new insertion-based construction heuristic for solving
        # the pickup and delivery problem with time windows. European Journal of Operational Research, 175(2),
        # 672–687. https://doi.org/10.1016/j.ejor.2005.05.012

        beta = wait_j + max_shift_j
        predecessors_max_shift_delta = dt.timedelta(0)
        index = insertion_index - 1

        while True:
            if beta >= self.max_shift_sequence[index] or index == 0:
                break
            if index == insertion_index - 1:
                predecessors_max_shift_delta = self.max_shift_sequence[index] - beta
            elif self.wait_duration_sequence[index + 1] > dt.timedelta(0):
                predecessors_max_shift_delta += min(
                    self.max_shift_sequence[index] - beta,
                    self.wait_duration_sequence[index + 1],
                )
            beta += self.wait_duration_sequence[index]
            index -= 1

        # [3] delta in max_shift of insertion_vertex itself
        vertex_max_shift_delta = (
            request.tw_close - max(arrival_j, request.tw_open) - max_shift_j
        )

        # [4] delta in max_shift of succeeding vertices, which is exactly the travel time delta
        successors_max_shift_delta = (
            instance.travel_duration([predecessor], [request])
            + instance.travel_duration([request], [successor])
            - instance.travel_duration([predecessor], [successor])
        )

        return (
            predecessors_max_shift_delta
            + vertex_max_shift_delta
            + successors_max_shift_delta
        )

    def insert_max_shift_delta(
        self, instance, insertion_indices: list[int], requests: list[Request]
    ):
        """
        returns the waiting time and max_shift time that would be assigned to insertion_vertices if they were
        inserted before insertion_indices
        """
        if len(insertion_indices) == 1:
            return self._single_insert_max_shift_delta(
                instance, insertion_indices[0], requests[0]
            )

        else:
            # sanity check whether insertion positions are sorted in ascending order
            assert all(
                insertion_indices[i] < insertion_indices[i + 1]
                for i in range(len(insertion_indices) - 1)
            )

            # create a temporary copy
            copy = deepcopy(self)

            # check all insertions sequentially
            total_max_shift_delta = dt.timedelta(0)
            for idx, (insertion_index, insertion_vertex) in enumerate(
                zip(insertion_indices, requests)
            ):
                max_shift_delta = copy._single_insert_max_shift_delta(
                    instance, insertion_index, insertion_vertex
                )
                total_max_shift_delta += max_shift_delta
                copy._single_insert_and_update(
                    instance, insertion_index, insertion_vertex
                )
            return total_max_shift_delta

    def plot(
        self,
        ax: plt.Axes = False,
        plot_requests: bool = True,
        plot_depot: bool = True,
        plot_stats: bool = True,
        color: str = "tab:red",
    ):
        if not ax:
            fig, ax = plt.subplots()
        # this is the depot
        depot: Depot = self._routing_sequence[0]
        if plot_depot:
            kwargs = dict(c=color, marker="*", zorder=3, s=250)
            ax.scatter(
                depot.x,
                depot.y,
                label="Depot",
                **kwargs,
            )

        # these are the requests
        request_x = [request.x for request in self._routing_sequence[1:-1]]
        request_y = [request.y for request in self._routing_sequence[1:-1]]
        if request_x:  # skip if the tour is empty
            if plot_requests:
                # plot the requests as dots
                ax.scatter(
                    request_x,
                    request_y,
                    c=color,
                    label=f"Tour {self.id_}",
                    zorder=3,
                    s=30,
                )
            # plot the lines between the requests
            ax.plot(request_x, request_y, color=color, linewidth=1, zorder=2)

            # Thin edges from and to the depot. The edge from the depot to the  first client is given an arrow head to
            # indicate route direction. We  don't do this for the edge returning to the depot because that adds a lot of
            # clutter at the depot.
            kwargs = dict(linewidth=0.5, color="grey")
            ax.plot(
                [request_x[-1], depot.x],
                [request_y[-1], depot.y],
                linewidth=0.25,
                color="grey",
            )
            ax.annotate(
                "",
                xy=(request_x[0], request_y[0]),
                xytext=(depot.x, depot.y),
                arrowprops=dict(arrowstyle="-|>", **kwargs),
                zorder=1,
            )

        # Add some statistics in a box: sum_travel_duration, sum_travel_distance, num_routing_stops
        if plot_stats:
            stats = self.summary()
            text = "\n".join(
                [
                    f"Tour ID: {stats['tour_id']}",
                    f"Stops: {stats['num_routing_stops']}",
                    f"Distance: {stats['sum_travel_distance']:.2f}",
                    f"Duration: {stats['sum_travel_duration']}",
                    f"Revenue: {stats['sum_revenue']}",
                ]
            )
            ax.text(
                0.05,
                0.95,
                text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.5),
            )

        # ax.grid(color="grey", linestyle="solid", linewidth=0.2)
        # ax.set_title("Solution")
        # ax.set_aspect("equal", "datalim")
        # ax.legend(frameon=False, ncol=2)
