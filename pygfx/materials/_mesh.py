from pyshader import Struct, vec4, vec2

from ..utils import array_from_shadertype
from ..datawrappers import Buffer
from ._base import Material

# todo: put in an example somewhere how to use storage buffers for vertex data:
# index: (pyshader.RES_INPUT, "VertexId", "i32")
# positions: (pyshader.RES_BUFFER, (1, 0), "Array(vec4)")
# position = positions[index]


class MeshBasicMaterial(Material):
    """ A material for drawing geometries in a simple shaded (flat or
    wireframe) way. This material is not affected by lights.
    """

    uniform_type = Struct(color=vec4, clim=vec2,)

    def __init__(self, **kwargs):
        super().__init__()

        self.uniform_buffer = Buffer(
            array_from_shadertype(self.uniform_type), usage="UNIFORM"
        )

        self._map = None
        self.color = 1, 1, 1, 1
        self.clim = 0, 1

        for argname, val in kwargs.items():
            if not hasattr(self, argname):
                raise AttributeError(f"No attribute '{argname}'")
            setattr(self, argname, val)

    @property
    def color(self):
        """ The uniform color of the mesh, as an rgba tuple.
        This value is ignored if a texture map is used.
        """
        return self.uniform_buffer.data["color"]

    @color.setter
    def color(self, color):
        self.uniform_buffer.data["color"] = color
        self.uniform_buffer.update_range(0, 1)
        self.dirty = True

    @property
    def map(self):
        """ The texture map specifying the color for each texture coordinate.
        """
        return self._map

    @map.setter
    def map(self, map):
        self._map = map
        self.dirty = True
        # todo: figure out a way for render funcs to tell when the pipelines that they create become invalid
        # but this code should not know about wgpu!
        self._wgpu_pipeline_dirty = True

    @property
    def clim(self):
        """ The contrast limits to apply to the map. Default (0, 1)
        """
        return self.uniform_buffer.data["clim"]

    @clim.setter
    def clim(self, clim):
        self.uniform_buffer.data["clim"] = clim
        self.uniform_buffer.update_range(0, 1)
        self.dirty = True


class MeshLambertMaterial(MeshBasicMaterial):
    """ A material for non-shiny surfaces, without specular highlights.
    """


class MeshPhongMaterial(MeshBasicMaterial):
    """ A material for shiny surfaces with specular highlights.
    """


class MeshStandardMaterial(MeshBasicMaterial):
    """ A standard physically based material, using Metallic-Roughness workflow.
    """


class MeshToonMaterial(MeshBasicMaterial):
    """ A standard physically based material, using Metallic-Roughness workflow.
    """
