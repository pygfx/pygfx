from ..utils import array_from_shadertype
from ..resources import Buffer
from ._base import Material


class MeshBasicMaterial(Material):
    """A material for drawing geometries in a simple shaded (flat or
    wireframe) way. This material is not affected by lights.
    """

    uniform_type = {
        "color": ("float32", (4,)),
        "clim": ("float32", (2,)),
    }

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

    def _wgpu_get_pick_info(self, pick_value):
        inst = pick_value[1]
        face = pick_value[2]
        coords = pick_value[3]
        coords = (coords & 0xFF0000) / 65536, (coords & 0xFF00) / 256, coords & 0xFF
        coords = coords[0] / 255, coords[1] / 255, coords[2] / 255
        return {"instance_index": inst, "face_index": face, "face_coords": coords}

    @property
    def color(self):
        """The uniform color of the mesh, as an rgba tuple.
        This value is ignored if a texture map is used.
        """
        return self.uniform_buffer.data["color"]

    @color.setter
    def color(self, color):
        self.uniform_buffer.data["color"] = color
        self.uniform_buffer.update_range(0, 1)

    @property
    def map(self):
        """The texture map specifying the color for each texture coordinate."""
        return self._map

    @map.setter
    def map(self, map):
        self._map = map

    @property
    def clim(self):
        """The contrast limits to apply to the map. Default (0, 1)"""
        return self.uniform_buffer.data["clim"]

    @clim.setter
    def clim(self, clim):
        self.uniform_buffer.data["clim"] = clim
        self.uniform_buffer.update_range(0, 1)


class MeshNormalMaterial(MeshBasicMaterial):
    """A material that maps the normal vectors to RGB colors."""


class MeshNormalLinesMaterial(MeshBasicMaterial):
    """A material that shows surface normals as lines sticking out of the mesh."""

    def _wgpu_get_pick_info(self, pick_value):
        return {}  # No picking for normal lines


# todo: MeshLambertMaterial(MeshBasicMaterial):
# A material for non-shiny surfaces, without specular highlights.


class MeshPhongMaterial(MeshBasicMaterial):
    """A material for shiny surfaces with specular highlights.

    The material uses a non-physically based Blinn-Phong model for
    calculating reflectance. Unlike the Lambertian model used in the
    MeshLambertMaterial this can simulate shiny surfaces with specular
    highlights (such as varnished wood).
    """


# todo: MeshStandardMaterial(MeshBasicMaterial):
# A standard physically based material, using Metallic-Roughness workflow.


# todo: MeshToonMaterial(MeshBasicMaterial):
# A cartoon-style mesh material.


class MeshSliceMaterial(MeshBasicMaterial):
    """A material that displays a slices of the mesh."""

    uniform_type = dict(
        color=("float32", 4),
        plane=("float32", 4),
        clim=("float32", 2),
        thickness=("float32",),
    )

    def __init__(self, plane=(0, 0, 1, 0), thickness=2.0, **kwargs):
        super().__init__(plane=plane, thickness=thickness, **kwargs)

    @property
    def plane(self):
        """The plane to slice at, represented with 4 floats ``(a, b, c, d)``,
        which make up the equation: ``ax + by + cz + d = 0`` The plane
        definition applies to the world space (of the scene).
        """
        return self.uniform_buffer.data["plane"]

    @plane.setter
    def plane(self, plane):
        self.uniform_buffer.data["plane"] = plane
        self.uniform_buffer.update_range(0, 1)

    @property
    def thickness(self):
        """The thickness of the line to draw the edge of the mesh."""
        return self.uniform_buffer.data["thickness"]

    @thickness.setter
    def thickness(self, thickness):
        self.uniform_buffer.data["thickness"] = thickness
        self.uniform_buffer.update_range(0, 1)
