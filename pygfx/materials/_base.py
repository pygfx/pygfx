from pyshader import Struct

from ..objects._base import ResourceContainer


class Material(ResourceContainer):

    uniform_type = Struct()

    def _wgpu_get_pick_info(self, pick_value):
        """Given a 2 or 4? element tuple, sampled from the
        pick texture, return info about what was picked in the object.
        """
        return {}
