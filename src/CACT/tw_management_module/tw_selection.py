import abc
import logging
import random

from core_module.instance import CAHDInstance
from core_module.request import Request
from tw_management_module.time_window import TimeWindow

logger = logging.getLogger(__name__)


class TWSelectionBehavior(abc.ABC):
    def __init__(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}"

    # NOTE maybe in the future, i have to store also time window preferences / tw selection behavior in the instance
    def execute(
        self, instance: CAHDInstance, tw_offer_set: list[TimeWindow], request: Request
    ):
        # may return False if no TW fits the preference
        if tw_offer_set:
            return self.select_tw(instance, tw_offer_set, request)
        # if the tw_offer_set is empty
        else:
            return False

    @abc.abstractmethod
    def select_tw(
        self, instance: CAHDInstance, tw_offer_set: list[TimeWindow], request: Request
    ) -> TimeWindow:
        pass


class NoTW(TWSelectionBehavior):
    def execute(
        self, instance: CAHDInstance, tw_offer_set: list[TimeWindow], request: Request
    ):
        inst_time_horizon = TimeWindow(
            instance.execution_time_start, instance.execution_time_end
        )
        assert tw_offer_set == inst_time_horizon
        return inst_time_horizon

    def select_tw(
        self, instance: CAHDInstance, tw_offer_set: list[TimeWindow], request: Request
    ) -> TimeWindow:
        pass


class UniformPreference(TWSelectionBehavior):
    """Will randomly select a TW"""

    def select_tw(
        self, instance: CAHDInstance, tw_offer_set: list[TimeWindow], request: Request
    ) -> TimeWindow:
        return random.choice(tw_offer_set)


class UnequalPreference(TWSelectionBehavior):
    """
    Following Köhler,C., Ehmke,J.F., & Campbell,A.M. (2020). Flexible time window management for attended home
    deliveries. Omega, 91, 102023. https://doi.org/10.1016/j.omega.2019.01.001

    Late time windows exhibit a much higher popularity and are requested by 90% of the customers.
    """

    def select_tw(
        self, instance: CAHDInstance, tw_offer_set: list[TimeWindow], request: Request
    ) -> TimeWindow:
        # preference can either be for early (10%) or late (90%) time windows
        pref = random.random()

        planning_horizon_midpoint = (
            instance.execution_time_start
            + (instance.execution_time_end - instance.execution_time_start) / 2
        )

        # early preference
        if pref <= 0.1:
            attractive_tws = [
                tw for tw in tw_offer_set if tw.open <= planning_horizon_midpoint
            ]
        # late preference
        else:
            attractive_tws = [
                tw for tw in tw_offer_set if tw.close >= planning_horizon_midpoint
            ]
        if attractive_tws:
            return random.choice(attractive_tws)
        else:
            return False


class EarlyPreference(TWSelectionBehavior):
    """Will always select the earliest TW available based on the time window opening"""

    def select_tw(
        self, instance: CAHDInstance, tw_offer_set: list[TimeWindow], request: Request
    ) -> TimeWindow:
        return min(tw_offer_set, key=lambda tw: tw.open)


class LatePreference(TWSelectionBehavior):
    """Will always select the latest TW available based on the time window closing"""

    def select_tw(
        self, instance: CAHDInstance, tw_offer_set: list[TimeWindow], request: Request
    ) -> TimeWindow:
        return max(tw_offer_set, key=lambda tw: tw.close)


# class PreferEarlyAndLate(TWSelectionBehavior):
#     def select_tw(self, tw_offer_set):
#         pass
