import datetime as dt

from core_module.vertex import Vertex


class Depot(Vertex):
    """
    Represents a depot in the instance. The depot is a node in the graph and has a location, and a time window
    """

    def __init__(
        self,
        label: str,
        vertex: int,
        x: float,
        y: float,
        tw_open: dt.datetime,
        tw_close: dt.datetime,
    ):
        self._index = label
        super().__init__(vertex, x, y)
        self.tw_open: dt.datetime = tw_open
        self.tw_close: dt.datetime = tw_close

        self.service_duration = dt.timedelta(0)

    def __repr__(self):
        return (
            f"Depot(label={self.label}, uid={self.uid}, x={self.x}, y={self.y}, "
            f"tw_open={self.tw_open}, tw_close={self.tw_close})"
        )

    def __hash__(self):
        return hash((self.label, self.uid, self.x, self.y))

    @property
    def label(self):
        return self._index

