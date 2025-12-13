class Vertex:
    """Vertex class. Defines a unique identifier and a two-dimenionsal coordinate."""

    def __init__(self, uid: int, x: float, y: float):
        self._uid = uid
        """It is in no way guaranteed that the uid is truly unique among all vertices.
         It is used to index e.g. distance matrices"""
        self.x = x
        self.y = y

    @property
    def uid(self):
        return self._uid

