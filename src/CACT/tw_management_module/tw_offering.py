import abc
import datetime as dt

from core_module.carrier import Carrier
from core_module.instance import CAHDInstance
from core_module.tour import Tour
from core_module.solution import CAHDSolution
from core_module.request import Request
from tw_management_module.time_window import TimeWindow
from utility_module import utils as ut
from utility_module.parameterized_class import ParameterizedClass


class TWOfferingBehavior(ParameterizedClass):
    def __init__(self, time_window_length: dt.timedelta):
        self.time_window_length = time_window_length
        self._params = {"time_window_length": time_window_length}

        self.time_windows = ut.generate_time_windows(time_window_length)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.params})"

    def execute_for_carrier(
        self, instance: CAHDInstance, carrier: Carrier, request: Request
    ):
        # make sure that the request has not been given a tw yet
        assert request.tw_open in (ut.EXECUTION_START_TIME, None), (
            f"tw_open={request.tw_open}"
        )
        assert request.tw_close in (ut.END_TIME, None), f"tw_close={request.tw_close}"

        tw_valuations = []
        for tw in self.time_windows:
            tw_valuations.append(
                self.evaluate_tw_for_carrier(instance, carrier, request, tw)
            )
        offered_time_windows = list(
            sorted(zip(tw_valuations, self.time_windows), key=lambda x: x[0])
        )
        offered_time_windows = [
            tw for valuation, tw in offered_time_windows if valuation >= 0
        ]
        return offered_time_windows

    def execute_central(
        self, instance: CAHDInstance, solution: CAHDSolution, request: Request
    ):
        """
        CentralPlanning perspective

        :param instance:
        :param solution:
        :param request:
        :return:
        """

        # make sure that the request has not been given a tw yet
        assert request.tw_open in (ut.EXECUTION_START_TIME, None), (
            f"tw_open={request.tw_open}"
        )
        assert request.tw_close in (ut.END_TIME, None), f"tw_close={request.tw_close}"
        tw_valuations = []
        for tw in self.time_windows:
            tw_valuations.append(
                self.evaluate_tw_central(instance, solution, request, tw)
            )
        offered_time_windows = list(
            sorted(zip(tw_valuations, self.time_windows), key=lambda x: x[0])
        )
        offered_time_windows = [
            tw for valuation, tw in offered_time_windows if valuation >= 0
        ]
        return offered_time_windows

    @abc.abstractmethod
    def evaluate_tw_for_carrier(
        self, instance: CAHDInstance, carrier: Carrier, request: Request, tw: TimeWindow
    ):
        pass

    @abc.abstractmethod
    def evaluate_tw_central(
        self,
        instance: CAHDInstance,
        solution: CAHDSolution,
        request: Request,
        tw: TimeWindow,
    ):
        pass


class FeasibleTW(TWOfferingBehavior):
    def evaluate_tw_for_carrier(
        self, instance: CAHDInstance, carrier: Carrier, request: Request, tw: TimeWindow
    ):
        """
        :return: 1 if TW is feasible, -1 else
        """
        old_tw_open = request.tw_open
        old_tw_close = request.tw_close
        # temporarily set the time window under consideration
        request.tw_open = tw.open
        request.tw_close = tw.close

        # can the carrier open a new pendulum tour and insert the request there?
        if len(carrier.tours) < instance.carriers_max_num_tours:
            tmp_tour = Tour("tmp", carrier.depot)
            if tmp_tour.insertion_feasibility_check(instance, [1], [request]):
                # undo the setting of the time window and return
                request.tw_open = old_tw_open
                request.tw_close = old_tw_close
                return 1

        # if no feasible new tour can be built, can the request be inserted into one of the existing tours?
        for tour in carrier.tours:
            for delivery_pos in range(1, len(tour)):
                if tour.insertion_feasibility_check(
                    instance, [delivery_pos], [request]
                ):
                    # undo the setting of the time window and return
                    request.tw_open = old_tw_open
                    request.tw_close = old_tw_close
                    return 1

        # undo the setting of the time window and return
        request.tw_open = old_tw_open
        request.tw_close = old_tw_close
        return -1

    def evaluate_tw_central(
        self,
        instance: CAHDInstance,
        solution: CAHDSolution,
        request: Request,
        tw: TimeWindow,
    ):
        """
        :return: 1 if TW is feasible, -1 else
        """

        # temporarily set the time window under consideration
        request.tw_open = tw.open
        request.tw_close = tw.close

        for carrier in solution.carriers:
            # can the carrier open a new pendulum tour and insert the request there?
            if len(carrier.tours) < instance.carriers_max_num_tours:
                tmp_tour = Tour("tmp", carrier.depot)
                if tmp_tour.insertion_feasibility_check(instance, [1], [request]):
                    # undo the setting of the time window and return
                    request.tw_open = ut.EXECUTION_START_TIME
                    request.tw_close = ut.END_TIME
                    return 1

            # if no feasible new tour can be built, can the request be inserted into one of the existing tours?
            for tour in carrier.tours:
                for pos in range(1, len(tour)):
                    if tour.insertion_feasibility_check(instance, [pos], [request]):
                        # undo the setting of the time window and return
                        request.tw_open = ut.EXECUTION_START_TIME
                        request.tw_close = ut.END_TIME
                        return 1

        # undo the setting of the time window and return
        request.tw_open = ut.EXECUTION_START_TIME
        request.tw_close = ut.END_TIME
        return -1


class NoTw(TWOfferingBehavior):
    def execute_central(
        self, instance: CAHDInstance, solution: CAHDSolution, request: Request
    ):
        return [ut.EXECUTION_TIME_HORIZON]

    def execute_for_carrier(
        self, instance: CAHDInstance, carrier: Carrier, request: Request
    ):
        return [ut.EXECUTION_TIME_HORIZON]

    def evaluate_tw_for_carrier(
        self, instance: CAHDInstance, carrier: Carrier, request: Request, tw: TimeWindow
    ):
        pass

    def evaluate_tw_central(
        self,
        instance: CAHDInstance,
        solution: CAHDSolution,
        request: Request,
        tw: TimeWindow,
    ):
        pass
