from ._base import Material


class MeshBasicMaterial(Material):
    """A material for drawing geometries in a simple shaded (flat or
    wireframe) way. This material is not affected by lights.
    """

    uniform_type = dict(
        color=("float32", 4),
        clipping_planes=("float32", (0, 1, 4)),  # array<vec4<f32>,N>
        clim=("float32", 2),
        opacity=("float32",),
    )

    def __init__(
        self,
        color=(1, 1, 1, 1),
        clim=(0, 1),
        map=None,
        side="BOTH",
        winding="CCW",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.color = color
        self.clim = clim
        self.map = map
        self.side = side
        self.winding = winding

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

    @property
    def winding(self):
        """The winding determines what is the front-facing side of a
        triangle. Possible values are "CW" and "CCW", meaning clock-wise
        and counter-clock-wise, respectively. By default this is CCW
        like in e.g. ThreeJS.
        """
        return self._winding

    @winding.setter
    def winding(self, value):
        winding = str(value).upper()
        if winding in ("CW", "CCW"):
            self._winding = winding
        else:
            raise ValueError(f"Unexpected winding: '{value}'")
        self._bump_rev()

    @property
    def side(self):
        """Defines which side of faces will be rendered - front, back
        or both. By default this is both. Setting to front or back will
        not render faces from that side, a feature also known as culling.
        """
        return self._side

    @side.setter
    def side(self, value):
        side = str(value).upper()
        if side in ("FRONT", "BACK", "BOTH"):
            self._side = side
        else:
            raise ValueError(f"Unexpected side: '{value}'")
        self._bump_rev()


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
        clipping_planes=("float32", (0, 1, 4)),  # array<vec4<f32>,N>
        clim=("float32", 2),
        thickness=("float32",),
        opacity=("float32",),
    )

    def __init__(self, plane=(0, 0, 1, 0), thickness=2.0, **kwargs):
        super().__init__(**kwargs)
        self.plane = plane
        self.thickness = thickness

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
