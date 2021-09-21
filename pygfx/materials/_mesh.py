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
        wireframe=("float32",),
    )

    def __init__(
        self,
        color=(1, 1, 1, 1),
        clim=(0, 1),
        map=None,
        wireframe=False,
        wireframe_thickness=1,
        side="BOTH",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.color = color
        self.clim = clim
        self.map = map
        self.wireframe = wireframe
        self.wireframe_thickness = wireframe_thickness
        self.side = side

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
        v1, v2 = self.uniform_buffer.data["clim"]
        return float(v1), float(v2)

    @clim.setter
    def clim(self, clim):
        self.uniform_buffer.data["clim"] = clim
        self.uniform_buffer.update_range(0, 1)

    @property
    def side(self):
        """Defines which side of faces will be rendered - front, back
        or both. By default this is "both". Setting to "front" or "back" will
        not render faces from that side, a feature also known as culling.

        Which side of the mesh is the front is determined by the winding of the faces.
        Counter-clockwise (CCW) winding is assumed. If this is not the case,
        adjust your geometry (using e.g. ``np.fliplr()`` on ``geometry.indices``).
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

    @property
    def wireframe(self):
        """Render geometry as a wireframe. Default is False (i.e. render as polygons)."""
        return self.uniform_buffer.data["wireframe"] > 0

    @wireframe.setter
    def wireframe(self, value):
        is_wiremode = bool(value)
        was_wiremode = self.uniform_buffer.data["wireframe"] > 0
        if was_wiremode == is_wiremode:
            return
        # Set uniform
        # We use a trick to make negative values indicate no-wireframe mode
        thickness = self.wireframe_thickness
        if is_wiremode:
            self.uniform_buffer.data["wireframe"] = thickness
        else:
            self.uniform_buffer.data["wireframe"] = -thickness
        # Trigger a pipleine rebuild if the mode changes
        self.uniform_buffer.update_range(0, 1)
        self._bump_rev()

    @property
    def wireframe_thickness(self):
        """The thickness of the lines when rendering as a wireframe."""
        return abs(float(self.uniform_buffer.data["wireframe"])) or 1

    @wireframe_thickness.setter
    def wireframe_thickness(self, value):
        value = max(0.01, float(value))
        if self.uniform_buffer.data["wireframe"] > 0:
            self.uniform_buffer.data["wireframe"] = value
        else:
            self.uniform_buffer.data["wireframe"] = -value
        self.uniform_buffer.update_range(0, 1)


# todo: MeshLambertMaterial? In ThreeJS this material uses Gouroud shading with the Lambertian light model.


class MeshPhongMaterial(MeshBasicMaterial):
    """A material affected by light, diffuse and with specular
    highlights. This material uses the Blinn-Phong reflectance model.
    If the specular color is turned off, Lambertian shading is obtained.
    """

    # For reference:
    #
    # Lambertion shading, or Lambertian reflection, is a model to
    # calculate the diffuse component of a lit surface. Using this by
    # itself produces a matte look. All the below use a Lambertion term.
    #
    # Gouraud shading means doing the light-math in the vertex shader
    # and interpolating the final color over the face, often resulting
    # in a somewhat "interpolated" look. Back in the day this mattered
    # for performance, but it's silly now.
    #
    # Phong shading means interpolating the normals and doing the
    # light-math for each fragment.
    #
    # The Phong reflection model refers to the combination of ambient,
    # diffuse and specular lights, and the way that these are
    # calculated.
    #
    # The Blinn-Phong reflection model, also called the modified Phong
    # reflection model, is a tweak to how the reflection is calculated,
    # using a halfway factor, that was intended mostly as a performance
    # optimization, but apparently is a more accurate approximation of
    # how light behaves, or so they say.
    #
    # Flat shading refers to using the same color for the whole face.
    # This is what you get if the geometry has indices that do not share
    # vertices. But we can also obtain it by calculating the face normal
    # using derivatives of the world pos.

    uniform_type = dict(
        color=("float32", 4),
        emissive_color=("float32", 4),
        clipping_planes=("float32", (0, 1, 4)),  # array<vec4<f32>,N>
        clim=("float32", 2),
        opacity=("float32",),
        wireframe=("float32",),
        shininess=("float32",),
    )

    def __init__(self, shininess=30, emissive=(0, 0, 0, 0), **kwargs):
        super().__init__(**kwargs)
        self.emissive = emissive
        self.shininess = shininess

    @property
    def emissive(self):
        """The emissive (light) color of the mesh, as an rgba tuple.
        This color is added to the final color and is unaffected by lighting.
        The alpha channel of this color is ignored.
        """
        return self.uniform_buffer.data["emissive_color"]

    @emissive.setter
    def emissive(self, color):
        self.uniform_buffer.data["emissive_color"] = color
        self.uniform_buffer.update_range(0, 1)

    @property
    def shininess(self):
        """How shiny the specular highlight is; a higher value gives a sharper highlight.
        Default is 30.
        """
        return float(self.uniform_buffer.data["shininess"])

    @shininess.setter
    def shininess(self, value):
        self.uniform_buffer.data["shininess"] = value
        self.uniform_buffer.update_range(0, 1)


class MeshFlatMaterial(MeshPhongMaterial):
    """A material that applies lighting per-face (non-interpolated).
    This gives a "pixelated" look, but can also be usefull if one wants
    to show the (size of) the triangle faces. The shading and
    reflectance model is the same as for ``MeshPhongMaterial``.
    """


# todo: MeshStandardMaterial(MeshBasicMaterial):
# A standard physically based material, using Metallic-Roughness workflow.


# todo: MeshToonMaterial(MeshBasicMaterial):
# A cartoon-style mesh material.


class MeshNormalMaterial(MeshBasicMaterial):
    """A material that maps the normal vectors to RGB colors."""


class MeshNormalLinesMaterial(MeshBasicMaterial):
    """A material that shows surface normals as lines sticking out of the mesh."""

    def _wgpu_get_pick_info(self, pick_value):
        return {}  # No picking for normal lines


class MeshSliceMaterial(MeshBasicMaterial):
    """A material that displays a slices of the mesh."""

    uniform_type = dict(
        color=("float32", 4),
        plane=("float32", 4),
        clipping_planes=("float32", (0, 1, 4)),  # array<vec4<f32>,N>
        clim=("float32", 2),
        thickness=("float32",),
        opacity=("float32",),
        wireframe=("float32",),  # Not used, but must be defined
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
