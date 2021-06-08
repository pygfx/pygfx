from ._base import Material
from ..utils import array_from_shadertype
from ..resources import Buffer


class PointsMaterial(Material):
    """The default material used by Points. Renders (antialiased) disks
    of the given size and color.
    """

    uniform_type = dict(color=("float32", 4), size=("float32",))

    def __init__(self, **kwargs):
        super().__init__()

        self.uniform_buffer = Buffer(
            array_from_shadertype(self.uniform_type), usage="UNIFORM"
        )

        self._map = None
        self.color = 1, 1, 1, 1
        self.size = 1

        for argname, val in kwargs.items():
            if not hasattr(self, argname):
                raise AttributeError(f"No attribute '{argname}'")
            setattr(self, argname, val)

    def _wgpu_get_pick_info(self, pick_value):
        # The instance is zero while renderer doesn't support instancing
        instance = pick_value[1]
        vertex = pick_value[2]
        return {"instance_index": instance, "vertex_index": vertex}

    @property
    def color(self):
        """The color of the points (if map is not set)."""
        return self.uniform_buffer.data["color"]

    @color.setter
    def color(self, color):
        self.uniform_buffer.data["color"] = tuple(color)
        self.uniform_buffer.update_range(0, 1)

    # @property
    # def map(self):
    #     """ The 1D texture map specifying the color for each point.
    #     """
    #     return self._map
    #
    # @map.setter
    # def map(self, map):
    #     self._map = map

    @property
    def size(self):
        """The size (diameter) of the points, in logical pixels."""
        return self.uniform_buffer.data["size"]

    @size.setter
    def size(self, size):
        self.uniform_buffer.data["size"] = size

    # todo: sizeAttenuation


class GaussianPointsMaterial(PointsMaterial):
    """A material for points, renders Gaussian blobs with a standard
    deviation of 1/6 of the size.
    """


# idea: a MarkerMaterial with more options for the shape, and an edge around the shape.
# Though perhaps such a material should be part of a higher level plotting lib.
