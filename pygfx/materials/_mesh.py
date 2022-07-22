import math
from ._base import Material
from ..resources import TextureView
from ..utils import unpack_bitfield
from ..utils.color import Color


class MeshBasicMaterial(Material):
    """A material for drawing geometries in a simple shaded (flat or
    wireframe) way. This material is not affected by lights.
    """

    uniform_type = dict(
        color="4xf4",
        wireframe="f4",
    )

    def __init__(
        self,
        color=(1, 1, 1, 1),
        vertex_colors=False,
        map=None,
        wireframe=False,
        wireframe_thickness=1,
        side="BOTH",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.color = color
        self.vertex_colors = bool(vertex_colors)
        self.map = map
        self.wireframe = wireframe
        self.wireframe_thickness = wireframe_thickness
        self.side = side

    def _wgpu_get_pick_info(self, pick_value):
        # This should match with the shader
        values = unpack_bitfield(
            pick_value, wobject_id=20, index=26, coord1=6, coord2=6, coord3=6
        )
        return {
            "face_index": values["index"],
            "face_coord": (
                values["coord1"] / 64,
                values["coord2"] / 64,
                values["coord3"] / 64,
            ),
        }

    @property
    def color(self):
        """The uniform color of the mesh, as an rgba tuple.
        This value is ignored if a texture map is used.
        """
        return Color(self.uniform_buffer.data["color"])

    @color.setter
    def color(self, color):
        color = Color(color)
        self.uniform_buffer.data["color"] = color
        self.uniform_buffer.update_range(0, 1)
        self._store.color_is_transparent = color.a < 1

    @property
    def color_is_transparent(self):
        """Whether the color is (semi) transparent (i.e. not fully opaque)."""
        return self._store.color_is_transparent

    @property
    def vertex_colors(self):
        """Whether to use the vertex colors provided in the geometry."""
        return self._store.vertex_colors

    @vertex_colors.setter
    def vertex_colors(self, value):
        self._store.vertex_colors = bool(value)

    @property
    def map(self):
        """The texture map specifying the color for each texture coordinate.
        The dimensionality of the map can be 1D, 2D or 3D, but should match the
        number of columns in the geometry's texcoords.
        """
        return self._store.map

    @map.setter
    def map(self, map):
        assert map is None or isinstance(map, TextureView)
        self._store.map = map

    @property
    def side(self):
        """Defines which side of faces will be rendered: "FRONT", "BACK", or "BOTH".
        By default this is "BOTH". Setting to "FRONT" or "BACK" will only render
        faces from that side, hiding the other. A feature also known as culling.

        Which side of the mesh is the front is determined by the winding of the faces.
        Counter-clockwise (CCW) winding is assumed. If this is not the case,
        adjust your geometry (using e.g. ``np.fliplr()`` on ``geometry.indices``).
        """
        return self._store.side

    @side.setter
    def side(self, value):
        side = str(value).upper()
        if side in ("FRONT", "BACK", "BOTH"):
            self._store.side = side
        else:
            raise ValueError(f"Unexpected side: '{value}'")

    @property
    def wireframe(self):
        """Render geometry as a wireframe. Default is False (i.e. render as polygons)."""
        return self._store.wireframe

    @wireframe.setter
    def wireframe(self, value):
        is_wiremode = bool(value)
        self._store.wireframe = is_wiremode
        # Set uniform
        # We use a trick to make negative values indicate no-wireframe mode
        thickness = self.wireframe_thickness
        if is_wiremode:
            self.uniform_buffer.data["wireframe"] = thickness
        else:
            self.uniform_buffer.data["wireframe"] = -thickness
        self.uniform_buffer.update_range(0, 1)

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
        emissive_color="4xf4",
        specular_color="4xf4",
        shininess="f4",
        flat_shading="i4",
    )

    def __init__(
        self,
        shininess=30,
        emissive="#000000",
        specular="#111111",
        flat_shading=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.emissive = emissive
        self.shininess = shininess
        self.specular = specular
        self.flat_shading = flat_shading

    @property
    def emissive(self):
        """The emissive (light) color of the mesh.
        This color is added to the final color and is unaffected by lighting.
        The alpha channel of this color is ignored.
        """
        return Color(self.uniform_buffer.data["emissive_color"])

    @emissive.setter
    def emissive(self, color):
        color = Color(color)
        self.uniform_buffer.data["emissive_color"] = color
        self.uniform_buffer.update_range(0, 1)

    @property
    def specular(self):
        """The specular (highlight) color of the mesh.
        Default is a Color set to #111111 (very dark grey)"""

        return Color(self.uniform_buffer.data["specular_color"])

    @specular.setter
    def specular(self, color):
        color = Color(color)
        self.uniform_buffer.data["specular_color"] = color
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

    @property
    def flat_shading(self):
        """Whether the mesh is rendered with flat shading.
        A material that applies lighting per-face (non-interpolated).
        This gives a "pixelated" look, but can also be usefull if one wants
        to show the (size of) the triangle faces.
        """
        return bool(self.uniform_buffer.data["flat_shading"])

    @flat_shading.setter
    def flat_shading(self, value: bool):
        self.uniform_buffer.data["flat_shading"] = value
        self.uniform_buffer.update_range(0, 1)

    # TODO: more advanced mproperties, Unified with "MeshStandardMaterial".


class MeshFlatMaterial(MeshPhongMaterial):
    """A material that applies lighting per-face (non-interpolated).
    This gives a "pixelated" look, but can also be usefull if one wants
    to show the (size of) the triangle faces. The shading and
    reflectance model is the same as for ``MeshPhongMaterial``.
    """


# todo: MeshToonMaterial(MeshBasicMaterial):
# A cartoon-style mesh material.


class MeshNormalMaterial(MeshBasicMaterial):
    """A material that maps the normal vectors to RGB colors."""


class MeshNormalLinesMaterial(MeshBasicMaterial):
    """A material that shows surface normals as lines sticking out of the mesh."""

    def _wgpu_get_pick_info(self, pick_value):
        return {}  # No picking for normal lines


class MeshSliceMaterial(MeshBasicMaterial):
    """A material that displays a slice of the mesh."""

    uniform_type = dict(
        plane="4xf4",
        thickness="f4",
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


class MeshStandardMaterial(MeshBasicMaterial):
    """A standard physically based material, using Metallic-Roughness workflow."""

    # Physically based rendering (PBR) has recently become the standard in many 3D applications,
    # it use a physically correct model instead of using approximations for the way in which light interacts with a surface.
    # Technical details of the approach can be found is this paper from Disney:
    # "https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf",
    # by Brent Burley.

    uniform_type = dict(
        emissive_color="4xf4",
        roughness="f4",
        metalness="f4",
        normal_scale="2xf4",
        light_map_intensity="f4",
        ao_map_intensity="f4",
        emissive_intensity="f4",
        env_map_intensity="f4",
        env_map_max_mip_level="f4",
        flat_shading="i4",
    )

    def __init__(
        self,
        emissive=(0, 0, 0, 0),
        flat_shading=False,
        metalness=0.0,
        roughness=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.emissive = emissive
        self.flat_shading = flat_shading

        self.roughness = roughness
        self.metalness = metalness

        self.roughness_map = None
        self.metalness_map = None

        self.light_map = None
        self.light_map_intensity = 1.0

        self.ao_map = None
        self.ao_map_intensity = 1.0

        self.emissive_intensity = 1.0
        self.emissive_map = None

        self.normal_map = None
        self.normal_scale = (1, 1)

        self.alpha_map = None

        self._env_map = None
        self.env_map_intensity = 1.0

        # TODO: more advanced properties

    @property
    def emissive(self):
        """The emissive (light) color of the mesh, as an rgba tuple.
        This color is added to the final color and is unaffected by lighting.
        The alpha channel of this color is ignored.
        """
        return Color(self.uniform_buffer.data["emissive_color"])

    @emissive.setter
    def emissive(self, color):
        color = Color(color)
        self.uniform_buffer.data["emissive_color"] = color
        self.uniform_buffer.update_range(0, 1)

    @property
    def emissive_intensity(self):
        """Intensity of the emissive light. Modulates the emissive color. Default is 1."""
        return self.uniform_buffer.data["emissive_intensity"]

    @emissive_intensity.setter
    def emissive_intensity(self, value):
        self.uniform_buffer.data["emissive_intensity"] = value
        self.uniform_buffer.update_range(0, 1)

    @property
    def emissive_map(self):
        """The emissive map color is modulated by the emissive color and the emissive intensity.
        If you have an emissive map, be sure to set the emissive color to something other than black.
         Default is None.
        """
        return self._store.emissive_map

    @emissive_map.setter
    def emissive_map(self, value):
        self._store.emissive_map = value

    @property
    def metalness(self):
        """How much the material is like a metal. Non-metallic materials such as wood or stone use 0.0, metallic use 1.0, with nothing (usually) in between.
        Default is 0.0. A value between 0.0 and 1.0 could be used for a rusty metal look. If metalness_map is also provided, both values are multiplied."""
        return float(self.uniform_buffer.data["metalness"])

    @metalness.setter
    def metalness(self, value):
        self.uniform_buffer.data["metalness"] = value
        self.uniform_buffer.update_range(0, 1)

    @property
    def metalness_map(self):
        """The blue channel of this texture is used to alter the metalness of the material."""
        return self._store.metalness_map

    @metalness_map.setter
    def metalness_map(self, value):
        self._store.metalness_map = value

    @property
    def roughness(self):
        """How rough the material is. 0.0 means a smooth mirror reflection, 1.0 means fully diffuse. Default is 1.0.
        If roughness_map is also provided, both values are multiplied."""
        return float(self.uniform_buffer.data["roughness"])

    @roughness.setter
    def roughness(self, value):
        self.uniform_buffer.data["roughness"] = value
        self.uniform_buffer.update_range(0, 1)

    @property
    def roughness_map(self):
        """The green channel of this texture is used to alter the roughness of the material."""
        return self._store.roughness_map

    @roughness_map.setter
    def roughness_map(self, value):
        self._store.roughness_map = value

    @property
    def normal_scale(self):
        """How much the normal map affects the material. Typical ranges are 0-1. Default is (1,1)."""
        return self.uniform_buffer.data["normal_scale"]

    @normal_scale.setter
    def normal_scale(self, value):
        self.uniform_buffer.data["normal_scale"] = value
        self.uniform_buffer.update_range(0, 1)

    @property
    def normal_map(self):
        """The texture to create a normal map.
        The RGB values affect the surface normal for each pixel fragment and change the way the color is lit.
        Normal maps do not change the actual shape of the surface, only the lighting.
        """
        return self._store.normal_map

    @normal_map.setter
    def normal_map(self, value):
        self._store.normal_map = value

    @property
    def light_map(self):
        """The light map. Default is None."""
        return self._store.light_map

    @light_map.setter
    def light_map(self, value):
        self._store.light_map = value

    @property
    def light_map_intensity(self):
        """Intensity of the baked light. Default is 1.0."""
        return self.uniform_buffer.data["light_map_intensity"]

    @light_map_intensity.setter
    def light_map_intensity(self, value):
        self.uniform_buffer.data["light_map_intensity"] = value
        self.uniform_buffer.update_range(0, 1)

    @property
    def ao_map(self):
        """The red channel of this texture is used as the ambient occlusion map. Default is None."""
        return self._store.ao_map

    @ao_map.setter
    def ao_map(self, value):
        self._store.ao_map = value

    @property
    def ao_map_intensity(self):
        """Intensity of the ambient occlusion effect. Default is 1.0 Zero is no occlusion effect."""
        return self.uniform_buffer.data["ao_map_intensity"]

    @ao_map_intensity.setter
    def ao_map_intensity(self, value):
        self.uniform_buffer.data["ao_map_intensity"] = value
        self.uniform_buffer.update_range(0, 1)

    @property
    def env_map_intensity(self):
        """Scales the effect of the environment map by multiplying its color."""
        return self.uniform_buffer.data["env_map_intensity"]

    @env_map_intensity.setter
    def env_map_intensity(self, value):
        self.uniform_buffer.data["env_map_intensity"] = value
        self.uniform_buffer.update_range(0, 1)

    @property
    def env_map(self):
        """The environment map. To ensure a physically correct rendering,
        you should only add cube environment maps which were prefilterd.
        We provide a built-in mipmap generation process by setting
        the "generate_mipmap" property of texture to True.
        Default is None."""
        return self._env_map

    @env_map.setter
    def env_map(self, env_map):
        self._env_map = env_map

        width, height, _ = env_map.texture.size
        max_level = math.floor(math.log2(max(width, height))) + 1
        self.uniform_buffer.data["env_map_max_mip_level"] = float(max_level)
        self.uniform_buffer.update_range(0, 1)

    @property
    def flat_shading(self):
        """Whether the mesh is rendered with flat shading.
        A material that applies lighting per-face (non-interpolated).
        This gives a "pixelated" look, but can also be usefull if one wants
        to show the (size of) the triangle faces.
        """
        return bool(self.uniform_buffer.data["flat_shading"])

    @flat_shading.setter
    def flat_shading(self, value: bool):
        self.uniform_buffer.data["flat_shading"] = value
        self.uniform_buffer.update_range(0, 1)
