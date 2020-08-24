from pyshader import Struct

from ..objects._base import TrackableObject


class Material(TrackableObject):

    uniform_type = Struct()
