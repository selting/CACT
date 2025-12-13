from core_module import instance as it, solution as slt
from core_module.carrier import Carrier
from core_module.request import Request
from customer_acceptance_module.acceptance_measure import CustomerAcceptanceMeasure
from tw_management_module import tw_offering as two, tw_selection as tws


class CustomerAcceptanceBehavior:
    def __init__(
        self,
        request_acceptance_attractiveness: CustomerAcceptanceMeasure,
        time_window_offering: two.TWOfferingBehavior,
        time_window_selection: tws.TWSelectionBehavior,
    ):
        self.request_acceptance_attractiveness = request_acceptance_attractiveness
        self.time_window_offering = time_window_offering
        self.time_window_selection = time_window_selection

        self.params = {
            "request_acceptance_attractiveness": request_acceptance_attractiveness,
            "time_window_offering": time_window_offering,
            "time_window_selection": time_window_selection,
        }

    def __repr__(self):
        return f"{self.__class__.__name__}({self.params})"

    def execute_for_carrier(
        self, instance: it.CAHDInstance, carrier: Carrier, request: Request
    ):
        offer_set = self.time_window_offering.execute_for_carrier(
            instance, carrier, request
        )

        if offer_set:  # no overbooking, regular acceptance criterion
            if self.request_acceptance_attractiveness.evaluate(
                instance, carrier, request
            ):
                if selected_tw := self.time_window_selection.select_tw(
                    offer_set, request
                ):
                    return "accept_feasible", selected_tw
                else:
                    return "reject_preference_mismatch", None
            else:
                return "reject_not_attractive", None

        else:
            raise NotImplementedError(
                "Removed overbooking on 2024-09-03, this else section should not be triggered"
            )

    def execute_central(
        self, instance: it.CAHDInstance, solution: slt.CAHDSolution, request: Request
    ):
        offer_set = self.time_window_offering.execute_central(
            instance, solution, request
        )
        if offer_set:  # no overbooking, regular acceptance criterion
            if self.request_acceptance_attractiveness.evaluate_central(
                instance, solution, request
            ):
                if selected_tw := self.time_window_selection.select_tw(
                    offer_set, request
                ):
                    return "accept_feasible", selected_tw
                else:
                    return "reject_preference_mismatch", None
            else:
                return "reject_not_attractive", None

        else:  # potential overbooking
            raise NotImplementedError(
                "removed overbooking on 2024-09-03, this else block should not be triggered"
            )
