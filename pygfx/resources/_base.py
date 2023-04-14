from ..utils.trackable import Trackable


STRUCT_FORMAT_ALIASES = {"c": "B", "l": "i", "L": "I"}

FORMAT_MAP = {
    "b": "i1",
    "B": "u1",
    "h": "i2",
    "H": "u2",
    "i": "i4",
    "I": "u4",
    "e": "f2",
    "f": "f4",
}


class Resource(Trackable):
    """Resource base class."""

    pass
