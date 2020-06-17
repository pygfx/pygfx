from pyshader import Struct, f32, vec4

from ._base import Material
from ..utils import array_from_shadertype
from ..datawrappers import Buffer


class PointsMaterial(Material):
    """ The default material used by Points. Renders (antialiased) disks
    of the given size and color.
    """

    uniform_type = Struct(color=vec4, size=f32)

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

    @property
    def color(self):
        """ The color of the points (if map is not set).
        """
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
    #     self.dirty = True
    #     # todo: figure out a way for render funcs to tell when the pipelines that they create become invalid
    #     # but this code should not know about wgpu!
    #     self._wgpu_pipeline_dirty = True

    @property
    def size(self):
        """ The size (diameter) of the points, in logical pixels.
        """
        return self.uniform_buffer.data["size"]

    @size.setter
    def size(self, size):
        self.uniform_buffer.data["size"] = size

    # todo: sizeAttenuation


class GaussianPointsMaterial(PointsMaterial):
    """ A material for points, renders Gaussian blobs with a standard
    deviation of 1/6 of the size.
    """


# idea: a MarkerMaterial with more options for the shape, and an edge around the shape.
# Though perhaps such a material should be part of a higher level plotting lib.
