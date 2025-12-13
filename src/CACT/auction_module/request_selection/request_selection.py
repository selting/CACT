import logging
from abc import abstractmethod

from core_module.instance import CAHDInstance
from core_module.request import Request
from core_module.tour import Tour
from utility_module.parameterized_class import ParameterizedClass

logger = logging.getLogger(__name__)


class RequestSelectionStrategy(ParameterizedClass):
    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    # def select_and_release(self, instance: it.CAHDInstance, solution: slt.CAHDSolution) -> Assignment:
    #     selected_requests = self._select_requests(instance, solution)
    #     for request, carrier in selected_requests.items():
    #         carrier.release_requests(instance, [request])
    #     return selected_requests
    #
    # @abstractmethod
    # def _select_requests(self, instance, solution) -> Assignment:
    #     pass

    @abstractmethod
    def __call__(self, instance: CAHDInstance, tours: list[Tour], k: int) -> [Request]:
        """

        :param instance:
        :param tours:
        :param k:
        :return:
        """
        # TODO: in the future, maybe this may also allow selecting customers that are not in any Tour, e.g. when over-
        #  booking is possible.
        pass
