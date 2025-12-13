import datetime as dt

from core_module.vertex import Vertex


class Request(Vertex):
    """
    Represents a request in the instance. The request is a node in the graph and has a location, a time window, a service
    duration, a revenue, a load, and a carrier assignment. The carrier assignment is the initial assignment and can be
    changed during the optimization process.

    Attributes:
        vertex_uid (int): The index of the vertex in the graph.
        label (int): The index of the request in the instance.
        index(int):
        x (float): The x-coordinate of the request.
        y (float): The y-coordinate of the request.
        initial_carrier_assignment (int): The initial carrier assignment of the request.
        disclosure_time (Union[dt.datetime, int]): The disclosure time of the request. Will be converted to datetime.
        revenue (float): The revenue of the request.
        load (float): The load of the request.
        service_duration (Union[dt.timedelta, float]): Service duration of the request. Will be converted to timedelta.
        tw_open (Union[dt.datetime, int]): The open time window of the request. Will be converted to datetime.
        tw_close (Union[dt.datetime, int]): The close time window of the request. Will be converted to datetime.
    """

    def __init__(self,
                 vertex_uid: int,
                 label: str,
                 index: int,
                 x: float,
                 y: float,
                 initial_carrier_assignment: int,
                 disclosure_time: dt.datetime,
                 revenue: float,
                 load: float,
                 service_duration: dt.timedelta,
                 tw_open: dt.datetime,
                 tw_close: dt.datetime):
        self.label: str = label
        super().__init__(vertex_uid, x, y)
        self._index: int = index
        self.initial_carrier_assignment: int = initial_carrier_assignment
        self.disclosure_time: dt.datetime = disclosure_time
        self.revenue: float = revenue
        self.load: float = load
        self.service_duration: dt.timedelta = service_duration
        self.tw_open: dt.datetime = tw_open
        self.tw_close: dt.datetime = tw_close

    def __repr__(self):
        return (f"Request(uid={self.uid}, index={self.index}, label={self.label}, x={self.x}, y={self.y}, "
                f"initial_carrier_assignment={self.initial_carrier_assignment}, disclosure_time={self.disclosure_time}, "
                f"revenue={self.revenue}, load={self.load}, service_duration={self.service_duration}, "
                f"tw_open={self.tw_open}, tw_close={self.tw_close})")

    def __hash__(self):
        return hash((self.index, self.uid, self.x, self.y, self.initial_carrier_assignment))

    def __lt__(self, other):
        return self.index < other.index

    def __le__(self, other):
        return self.index <= other.index

    def __gt__(self, other):
        return self.index > other.index

    def __ge__(self, other):
        return self.index >= other.index

    @property
    def index(self):
        return self._index

