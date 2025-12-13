import random

from auction_module.bundle_generation.bundle_based.bundle import Bundle
from core_module.carrier import Carrier
from core_module.request import Request


class Assignment(dict):
    """
    Dictionary that maps requests to carriers. Each request is assigned to exactly one carrier:
    assignment[request] = carrier.
    An assignment is not aware of carriers that are not assigned any requests!
    """

    def __init__(self, carriers: tuple[Carrier], **kwargs):
        self._carriers = carriers
        super().__init__(**kwargs)

    def __setitem__(self, key, value):
        if not isinstance(key, Request):
            raise ValueError(
                f"key must be of type Request, type {type(key)} was provided"
            )
        if not isinstance(value, Carrier):
            raise ValueError(
                f"value must be of type Carrier, type {type(value)} was provided"
            )
        if value not in self._carriers:
            raise ValueError(f"{value} is not a valid carrier")
        super().__setitem__(key, value)

    @classmethod
    def random(cls, carriers: tuple[Carrier], requests: tuple[Request]):
        """
        Returns a random assignment of requests to carriers. Carriers are identified by their id.
        :param carriers:
        """
        assignment = cls(carriers)
        for request in requests:
            assignment[request] = carriers[random.randint(0, len(carriers) - 1)]
        return assignment

    @classmethod
    def from_rgs(
        cls, carriers: tuple[Carrier], requests: tuple[Request], rgs: list[int]
    ):
        assignment = cls(carriers)
        for request, carrier_idx in zip(requests, rgs):
            assignment[request] = carriers[carrier_idx]
        return assignment

    def carrier_to_bundle(self) -> dict[Carrier, Bundle]:
        """
        Returns a dictionary that maps carriers to bundles.
        """
        carrier_to_request_list = {carrier: [] for carrier in self._carriers}
        for request, carrier in self.items():
            carrier_to_request_list[carrier].append(request)

        carrier_to_bundle = dict()
        for carrier, bundle in carrier_to_request_list.items():
            carrier_to_bundle[carrier] = Bundle(tuple(sorted(self.requests())), bundle)
        return carrier_to_bundle

    def bundle_to_carrier(self) -> dict[Bundle, Carrier]:
        """
        Returns a dictionary that maps bundles to carriers.
        """
        return {bundle: carrier for carrier, bundle in self.carrier_to_bundle().items()}

    def binary(self) -> tuple[tuple[int]]:
        binary = []
        for carrier in sorted(self._carriers, key=lambda c: c.id_):
            binary.append(self.carrier_to_bundle()[carrier].bitstring)
        return tuple(binary)

    def as_rgs(self) -> tuple[int, ...]:
        """
        restricted growth string
        :return:
        """
        binary = self.binary()
        n = len(self.requests())
        rgs = [-1 for _ in range(n)]
        for carrier_idx, binary_bundle in enumerate(sorted(binary, reverse=True)):
            for j, item in enumerate(binary_bundle):
                if item:
                    rgs[j] = carrier_idx
        return tuple(rgs)

    def bundles(self) -> tuple[Bundle, ...]:
        return tuple(self.carrier_to_bundle().values())

    def requests(self):
        return tuple(self.keys())
