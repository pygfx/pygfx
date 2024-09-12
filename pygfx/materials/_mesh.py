import math
from ._base import Material
from ..resources import Texture
from ..utils import logger
from ..utils.color import Color
from ..utils.enums import ColorMode, VisibleSide


class MeshAbstractMaterial(Material):
    """Abstract mesh material.

    The abstract parent class for all mesh materials, defining their common properties.

    Parameters
    ----------
    color : Color
        The uniform color of the mesh (used depending on the ``color_mode``).
    color_mode : str | ColorMode
        The mode by which the mesh is coloured. Default 'auto'.
    map : Texture
        The texture map specifying the color at each texture coordinate. Optional.
    map_interpolation: str
        The method to interpolate the color map. Either 'nearest' or 'linear'. Default 'linear'.
    side : str | VisibleSide
        What side of the mesh is visible. Default "both".
    kwargs : Any
        Additional kwargs will be passed to the :class:`material base class
        <pygfx.Material>`.

    Notes
    -----
    The color format of the map is assumed to be sRGB. To use physical space
    instead, set the texture's colorspace property to "physical". To learn more
    about this, check out the :ref:`colorspace documentation <colorspaces>`

    The direction of a face is determined using Counter-clockwise (CCW) winding;
    i.e., if the fingers of your curled hand match the direction in which the
    face's vertices are defined then your thumb points into the "front"
    direction of the face. If this is not the case for your mesh, adjust its
    geometry (using e.g. ``np.fliplr()`` on ``geometry.indices``).

    """

    uniform_type = dict(
        Material.uniform_type,
        color="4xf4",
        wireframe="f4",
    )

    def __init__(
        self,
        color="#fff",
        color_mode="auto",
        map=None,
        map_interpolation="linear",
        side="both",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.color = color
        self.color_mode = color_mode
        self.map = map
        self.map_interpolation = map_interpolation
        self.side = side

    @property
    def color(self):
        """The uniform color of the mesh.
        This value is ignored if a texture map is used.
        """
        return Color(self.uniform_buffer.data["color"])

    @color.setter
    def color(self, color):
        color = Color(color)
        self.uniform_buffer.data["color"] = color
        self.uniform_buffer.update_full()
        self._store.color_is_transparent = color.a < 1

    @property
    def color_is_transparent(self):
        """Whether the color is (semi) transparent (i.e. not fully opaque)."""
        return self._store.color_is_transparent

    @property
    def color_mode(self):
        """The way that color is applied to the mesh.

        See :obj:`pygfx.utils.enums.ColorMode`:
        """
        # todo: does 'auto' take the presence of texcoords into account?
        return self._store.color_mode

    @color_mode.setter
    def color_mode(self, value):
        value = value or "auto"
        if value not in ColorMode:
            raise ValueError(
                f"MeshMaterial.color_mode must be a string in {ColorMode}, not {repr(value)}"
            )
        self._store.color_mode = value

    @property
    def vertex_colors(self):
        return self.color_mode == ColorMode.vertex

    @vertex_colors.setter
    def vertex_colors(self, value):
        raise DeprecationWarning(
            "vertex_colors is deprecated, use ``color_mode='vertex'``"
        )

    @property
    def map(self):
        """The texture map specifying the color for each texture coordinate.
        The dimensionality of the map can be 1D, 2D or 3D, but should
        match the number of columns in the geometry's texcoords.

        The colors in the map are assumed to be in sRGB space. To use
        physical space instead, set the texture's colorspace property
        to "physical".
        """
        return self._store.map

    @map.setter
    def map(self, map):
        assert map is None or isinstance(map, Texture)
        self._store.map = map

    @property
    def map_interpolation(self):
        """The method to interpolate the colormap. Either 'nearest' or 'linear'."""
        return self._store.map_interpolation

    @map_interpolation.setter
    def map_interpolation(self, value):
        assert value in ("nearest", "linear")
        self._store.map_interpolation = value

    @property
    def side(self):
        """Defines which side of faces will be rendered.

        See :obj:`pygfx.utils.enums.VisibleSide`:

        Which side of the mesh is the front is determined by the winding of the faces.
        Counter-clockwise (CCW) winding is assumed. If this is not the case,
        adjust your geometry (using e.g. ``np.fliplr()`` on ``geometry.indices``).
        """
        return self._store.side

    @side.setter
    def side(self, value):
        value = (value or "both").lower()
        if value not in VisibleSide:
            raise ValueError(
                f"MeshMaterial.side must be a string in {VisibleSide}, not {repr(value)}"
            )
        self._store.side = value


class MeshBasicMaterial(MeshAbstractMaterial):
    """Basic mesh material.

    A material for drawing geometries in a simple shaded (flat or wireframe)
    way. This material is not affected by lights.

    Parameters
    ----------
    env_map : Texture
        The environment map.
    wireframe : bool
        If True, render geometry as a wireframe, i.e., only render edges.
    wireframe_thickness : int
        The thickness of a rendered edge in screen pixels.
    flat_shading : bool
        If True, the shader will ignore the geometry's normal data and instead
        use face normals during lighting calculations.
    reflectivity : float
        How much the environment map affects the surface. also see ``env_combine_mode``.
        The default value is 1 and the valid range is between 0 (no reflections) and 1 (full reflections).
    refraction_ratio : float
        The index of refraction (IOR) of air (approximately 1) divided by the index of refraction of the material.
        It is used with ``env_mapping_mode`` set to "REFRACTION".
    env_combine_mode: str
        How the environment map affects the surface.
        The default value is "MULTIPLY" and the valid values are "MULTIPLY", "MIX", and "ADD".
    env_mapping_mode : str
        The environment mapping mode.
        The default value is "CUBE-REFLECTION" and the valid values are "CUBE-REFLECTION" and "CUBE-REFRACTION".
    kwargs : Any
        Additional kwargs will be passed to the :class:`base class <pygfx.MeshAbstractMaterial>`.
    """

    uniform_type = dict(
        MeshAbstractMaterial.uniform_type,
        reflectivity="f4",
        refraction_ratio="f4",
        light_map_intensity="f4",
        ao_map_intensity="f4",
    )

    def __init__(
        self,
        env_map=None,
        wireframe=False,
        wireframe_thickness=1,
        flat_shading=False,
        reflectivity=1.0,
        refraction_ratio=0.98,
        env_combine_mode="MULTIPLY",
        env_mapping_mode="CUBE-REFLECTION",
        light_map=None,
        ao_map=None,
        specular_map=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.wireframe = wireframe
        self.wireframe_thickness = wireframe_thickness
        self.flat_shading = flat_shading

        self.env_map = env_map
        self.env_combine_mode = env_combine_mode
        self.env_mapping_mode = env_mapping_mode
        self.reflectivity = reflectivity
        self.refraction_ratio = refraction_ratio

        self.light_map = light_map
        self.light_map_intensity = 1.0

        self.ao_map = ao_map
        self.ao_map_intensity = 1.0

        self.specular_map = specular_map

    @property
    def env_map(self):
        """The environment map."""
        return self._env_map

    @env_map.setter
    def env_map(self, env_map):
        self._env_map = env_map

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
        self.uniform_buffer.update_full()

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
        self.uniform_buffer.update_full()

    @property
    def flat_shading(self):
        """Whether the mesh is rendered with flat shading. When true,
        the shader will apply per-face surface normals, resulting in
        per-face lighting and a "pixelated", non-interpolated look,
        which can be useful to show the (size of) the triangle faces,
        or simply for the retro appearance. Note that the face normals
        are calculated from the vertex positions, ignoring the normal
        data in the geometry.
        """
        return self._store.flat_shading

    @flat_shading.setter
    def flat_shading(self, value: bool):
        self._store.flat_shading = bool(value)

    @property
    def reflectivity(self):
        """How much the environment map affects the surface. also see ``env_combine_mode``.
        The default value is 1 and the valid range is between 0 (no reflections) and 1 (full reflections).
        """
        return float(self.uniform_buffer.data["reflectivity"])

    @reflectivity.setter
    def reflectivity(self, value):
        self.uniform_buffer.data["reflectivity"] = value
        self.uniform_buffer.update_full()

    @property
    def refraction_ratio(self):
        """The index of refraction (IOR) of air (approximately 1) divided by the index of refraction of the material.
        It is used with ``env_mapping_mode`` set to "REFRACTION".
        """
        return float(self.uniform_buffer.data["refraction_ratio"])

    @refraction_ratio.setter
    def refraction_ratio(self, value):
        self.uniform_buffer.data["refraction_ratio"] = value
        self.uniform_buffer.update_full()

    @property
    def env_combine_mode(self):
        """How the environment map affects the surface.
        The default value is "MULTIPLY" and the valid values are "MULTIPLY", "MIX", and "ADD".
        """
        return self._store.env_combine_mode

    @env_combine_mode.setter
    def env_combine_mode(self, value):
        value = str(value).upper()
        if value in ("MULTIPLY", "MIX", "ADD"):
            self._store.env_combine_mode = value
        else:
            raise ValueError(f"Unexpected env_combine_mode: '{value}'")

    @property
    def env_mapping_mode(self):
        """The environment mapping mode.
        The default value is "REFLECTION" and the valid values are "CUBE-REFLECTION" and "CUBE-REFRACTION".
        """
        # todo: add support for other mapping modes,
        # "SPHERE-REFLECTION", "EQUIRECTANGULAR-REFLECTION", "EQUIRECTANGULAR-REFRACTION" etc.
        return self._store.env_mapping_mode

    @env_mapping_mode.setter
    def env_mapping_mode(self, value):
        value = str(value).upper()
        if value in ("CUBE-REFLECTION", "CUBE-REFRACTION"):
            self._store.env_mapping_mode = value
        else:
            raise ValueError(f"Unexpected env_mapping_mode: '{value}'")

    @property
    def light_map(self):
        """The light map to define pre-baked lighting (in srgb). Default is None.
        It requires a second set of texture coordinates (geometry.texcoords1)."""
        return self._store.light_map

    @light_map.setter
    def light_map(self, map):
        if map is not None and not isinstance(map, Texture):
            raise ValueError(
                f"light_map must be a Texture or None, received: {type(map)}"
            )
        self._store.light_map = map

    @property
    def light_map_intensity(self):
        """Intensity of the baked light. Scaling occurs in the physical
        color space. Default is 1.0.
        """
        return float(self.uniform_buffer.data["light_map_intensity"])

    @light_map_intensity.setter
    def light_map_intensity(self, value):
        self.uniform_buffer.data["light_map_intensity"] = value
        self.uniform_buffer.update_full()

    @property
    def ao_map(self):
        """The red channel of this texture is used as the ambient occlusion map. Default is None.
        It requires a second set of texture coordinates (geometry.texcoords1)."""
        return self._store.ao_map

    @ao_map.setter
    def ao_map(self, map):
        if map is not None and not isinstance(map, Texture):
            raise ValueError(f"ao_map must be a Texture or None, received: {type(map)}")
        self._store.ao_map = map

    @property
    def ao_map_intensity(self):
        """Intensity of the ambient occlusion effect. Default is 1.0, zero is no occlusion effect."""
        return float(self.uniform_buffer.data["ao_map_intensity"])

    @ao_map_intensity.setter
    def ao_map_intensity(self, value):
        self.uniform_buffer.data["ao_map_intensity"] = value
        self.uniform_buffer.update_full()

    @property
    def specular_map(self):
        """The specular map. Default is None."""
        return self._store.specular_map

    @specular_map.setter
    def specular_map(self, map):
        if map is not None and not isinstance(map, Texture):
            raise ValueError(
                f"specular_map must be a Texture or None, received: {type(map)}"
            )
        self._store.specular_map = map


class MeshPhongMaterial(MeshBasicMaterial):
    """Phong mesh material.

    A material affected by light, diffuse and with specular
    highlights. This material uses the Blinn-Phong reflectance model.
    If the specular color is turned off, Lambertian shading is obtained.

    Parameters
    ----------
    shininess : int
        How shiny the specular highlight is; a higher value gives a sharper
        highlight.
    emissive : Color
        The emissive (light) color of the mesh. This color is added to the final
        color and is unaffected by lighting. The alpha channel of this color is
        ignored.
    specular : Color
        The specular (highlight) color of the mesh.
    kwargs : Any
        Additional kwargs will be passed to the :class:`base class
        <pygfx.MeshBasicMaterial>`.

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
        MeshBasicMaterial.uniform_type,
        emissive_color="4xf4",
        specular_color="4xf4",
        normal_scale="2xf4",
        shininess="f4",
        emissive_intensity="f4",
    )

    def __init__(
        self,
        shininess=30,
        emissive="#000",
        specular="#494949",  # as physical: #111, the default in ThreeJS
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.emissive = emissive
        self.shininess = shininess
        self.specular = specular

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
        self.uniform_buffer.update_full()

    @property
    def specular(self):
        """The specular (highlight) color of the mesh."""

        return Color(self.uniform_buffer.data["specular_color"])

    @specular.setter
    def specular(self, color):
        color = Color(color)
        self.uniform_buffer.data["specular_color"] = color
        self.uniform_buffer.update_full()

    @property
    def shininess(self):
        """How shiny the specular highlight is; a higher value gives a sharper highlight.
        Default is 30.
        """
        return float(self.uniform_buffer.data["shininess"])

    @shininess.setter
    def shininess(self, value):
        self.uniform_buffer.data["shininess"] = value
        self.uniform_buffer.update_full()

    @property
    def emissive_map(self):
        """The emissive map color is modulated by the emissive color
        and the emissive intensity. If you have an emissive map, be
        sure to set the emissive color to something other than black.
        Note that both emissive color and emissive map are considered
        in srgb colorspace. Default None.
        """
        return self._store.emissive_map

    @emissive_map.setter
    def emissive_map(self, map):
        assert map is None or isinstance(map, Texture)
        self._store.emissive_map = map

    @property
    def emissive_intensity(self):
        """Intensity of the emissive light. Modulates the emissive color
        and emissive map. Default is 1.

        Note that the intensity is applied in the physical colorspace.
        You can think of it as scaling the number of photons. Therefore
        using an intensity of 0.5 is not the same as halving the
        emissive color, which is in srgb space.
        """
        return self.uniform_buffer.data["emissive_intensity"]

    @emissive_intensity.setter
    def emissive_intensity(self, value):
        self.uniform_buffer.data["emissive_intensity"] = value
        self.uniform_buffer.update_full()

    @property
    def normal_scale(self):
        """How much the normal map affects the material. This 2-tuple
        is multiplied with the normal_map's xy components (z is
        unaffected). Typical ranges are 0-1. Default is (1,1).
        """
        return tuple(self.uniform_buffer.data["normal_scale"])

    @normal_scale.setter
    def normal_scale(self, value):
        self.uniform_buffer.data["normal_scale"] = value
        self.uniform_buffer.update_range(0, 1)

    @property
    def normal_map(self):
        """The texture to create a normal map. Affects the surface
        normal for each pixel fragment and change the way the color is
        lit. Normal maps do not change the actual shape of the surface,
        only the lighting.
        """
        return self._store.normal_map

    @normal_map.setter
    def normal_map(self, map):
        assert map is None or isinstance(map, Texture)
        self._store.normal_map = map


class MeshToonMaterial(MeshBasicMaterial):
    """
    A material implementing toon shading.

    Parameters
    ----------
    emissive : Color
        The emissive (light) color of the mesh. This color is added to the final
        color and is unaffected by lighting. The alpha channel of this color is
        ignored.

    gradient_map : Texture
        Gradient map for toon shading. The gradient map sampler method is always 'nearest'.

    kwargs : Any
        Additional kwargs will be passed to the :class:`base class
        <pygfx.MeshBasicMaterial>`.
    """

    uniform_type = dict(
        MeshBasicMaterial.uniform_type,
        emissive_color="4xf4",
        normal_scale="2xf4",
        emissive_intensity="f4",
    )

    def __init__(
        self,
        emissive="#000",
        gradient_map=None,
        emissive_intensity=1.0,
        emissive_map=None,
        normal_map=None,
        normal_scale=(1, 1),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.emissive = emissive
        self.emissive_map = emissive_map
        self.emissive_intensity = emissive_intensity

        self.normal_map = normal_map
        self.normal_scale = normal_scale

        self.gradient_map = gradient_map

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
    def emissive_intensity(self):
        """Intensity of the emissive light. Modulates the emissive color
        and emissive map. Default is 1.

        Note that the intensity is applied in the physical colorspace.
        You can think of it as scaling the number of photons. Therefore
        using an intensity of 0.5 is not the same as halving the
        emissive color, which is in srgb space.
        """
        return float(self.uniform_buffer.data["emissive_intensity"])

    @emissive_intensity.setter
    def emissive_intensity(self, value):
        self.uniform_buffer.data["emissive_intensity"] = value
        self.uniform_buffer.update_range(0, 1)

    @property
    def normal_scale(self):
        """How much the normal map affects the material. This 2-tuple
        is multiplied with the normal_map's xy components (z is
        unaffected). Typical ranges are 0-1. Default is (1,1).
        """
        return tuple(self.uniform_buffer.data["normal_scale"])

    @normal_scale.setter
    def normal_scale(self, value):
        self.uniform_buffer.data["normal_scale"] = value
        self.uniform_buffer.update_range(0, 1)

    @property
    def normal_map(self):
        """The texture to create a normal map. Affects the surface
        normal for each pixel fragment and change the way the color is
        lit. Normal maps do not change the actual shape of the surface,
        only the lighting.
        """
        return self._store.normal_map

    @normal_map.setter
    def normal_map(self, map):
        assert map is None or isinstance(map, Texture)
        self._store.normal_map = map

    @property
    def gradient_map(self):
        """Gradient map for toon shading.
        The gradient map sampler method is always 'nearest'.
        """
        return self._store.gradient_map

    @gradient_map.setter
    def gradient_map(self, map):
        assert map is None or isinstance(map, Texture)
        self._store.gradient_map = map


class MeshNormalMaterial(MeshAbstractMaterial):
    """Color from Mesh normals.

    A material that maps the normal vectors to RGB colors.
    The ``flat_shading`` property can be used to show face normals.
    """


class MeshNormalLinesMaterial(MeshAbstractMaterial):
    """Render surface normals as lines.

    A material that shows surface normals as simple lines. The lines
    stick out from the vertices at the front faces by default.

    Parameters
    ----------
    line_length : float
        The length of the lines that indicate the normals, in local
        space. Set this to a negative value to make the lines stick out
        from the back faces.
    kwargs : Any
        Additional kwargs will be passed to the :class:`base class
        <pygfx.MeshBasicMaterial>`.

    """

    uniform_type = dict(
        MeshAbstractMaterial.uniform_type,
        line_length="f4",
    )

    def __init__(self, line_length=1.0, **kwargs):
        super().__init__(**kwargs)
        self.line_length = line_length

    def _wgpu_get_pick_info(self, pick_value):
        return {}  # No picking for normal lines

    @property
    def line_length(self):
        """The length of the lines that indicate the normals, in local
        space. Set this to a negative value to make the lines stick out
        from the back faces.
        """
        return float(self.uniform_buffer.data["line_length"])

    @line_length.setter
    def line_length(self, value):
        self.uniform_buffer.data["line_length"] = value
        self.uniform_buffer.update_full()


class MeshSliceMaterial(MeshAbstractMaterial):
    """Display a mesh slice.

    Parameters
    ----------
    plane : tuple
        The plane to slice at, represented with 4 floats ``(a, b, c, d)``, which
        make up the equation: ``ax + by + cz + d = 0`` The plane definition
        applies to the world space (of the scene).
    thickness : float
        The thickness of the line to draw the edge of the mesh in screen space
        (px).
    kwargs : Any
        Additional kwargs will be passed to the :class:`base class
        <pygfx.MeshBasicMaterial>`.

    """

    uniform_type = dict(
        MeshAbstractMaterial.uniform_type,
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
        return tuple(self.uniform_buffer.data["plane"])

    @plane.setter
    def plane(self, plane):
        self.uniform_buffer.data["plane"] = plane
        self.uniform_buffer.update_full()

    @property
    def thickness(self):
        """The thickness of the line to draw the edge of the mesh."""
        return float(self.uniform_buffer.data["thickness"])

    @thickness.setter
    def thickness(self, thickness):
        self.uniform_buffer.data["thickness"] = thickness
        self.uniform_buffer.update_full()


class MeshStandardMaterial(MeshBasicMaterial):
    """PBR shaded material.

    A standard physically based material, applying PBR (Physically based rendering)
    using the Metallic-Roughness workflow.

    Parameters
    ----------
    emissive : Color
        The emissive color of the mesh. I.e. the color that the object emits
        even when not lit by a light source. This color is added to the final
        color and unaffected by lighting. The alpha channel is ignored.
    metalness : float
        How much the material looks like a metal. Non-metallic materials such as
        wood or stone use 0.0, metal use 1.0, with nothing (usually) in between.
        Default is 0.0. A value between 0.0 and 1.0 could be used for a rusty
        metal look. If metalness_map is also provided, both values are
        multiplied.
    roughness : float
        How rough the material is. 0.0 means a smooth mirror reflection, 1.0
        means fully diffuse. Default is 1.0. If roughness_map is also provided,
        both values are multiplied.
    kwargs : Any
        Additional kwargs will be passed to the :class:`base class
        <pygfx.MeshBasicMaterial>`.

    """

    # Physically based rendering (PBR) has recently become the standard
    # in many 3D applications, it use a physically correct model instead
    # of using approximations for the way in which light interacts with
    # a surface. Technical details of the approach can be found is this
    # paper from Disney (by Brent Burley):
    # https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf

    uniform_type = dict(
        MeshBasicMaterial.uniform_type,
        emissive_color="4xf4",
        roughness="f4",
        metalness="f4",
        normal_scale="2xf4",
        emissive_intensity="f4",
        env_map_intensity="f4",
        env_map_max_mip_level="f4",
    )

    def __init__(
        self,
        emissive="#000",
        metalness=0.0,
        roughness=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.emissive = emissive
        self.roughness = roughness
        self.metalness = metalness

        self.roughness_map = None
        self.metalness_map = None

        self.emissive_map = None
        self.emissive_intensity = 1.0

        self.normal_map = None
        self.normal_scale = (1, 1)

        self.env_map_intensity = 1.0

        # Note: there are more advanced properties to add, e.g. displacement_map, alpha_map

    @property
    def emissive(self):
        """The emissive color of the mesh. I.e. the color that the
        object emits even when not lit by a light source. This color
        is added to the final color and unaffected by lighting. The
        alpha channel is ignored.
        """
        return Color(self.uniform_buffer.data["emissive_color"])

    @emissive.setter
    def emissive(self, color):
        color = Color(color)
        self.uniform_buffer.data["emissive_color"] = color
        self.uniform_buffer.update_full()

    @property
    def emissive_map(self):
        """The emissive map color is modulated by the emissive color
        and the emissive intensity. If you have an emissive map, be
        sure to set the emissive color to something other than black.
        Note that both emissive color and emissive map are considered
        in srgb colorspace. Default None.
        """
        return self._store.emissive_map

    @emissive_map.setter
    def emissive_map(self, map):
        assert map is None or isinstance(map, Texture)
        self._store.emissive_map = map

    @property
    def emissive_intensity(self):
        """Intensity of the emissive light. Modulates the emissive color
        and emissive map. Default is 1.

        Note that the intensity is applied in the physical colorspace.
        You can think of it as scaling the number of photons. Therefore
        using an intensity of 0.5 is not the same as halving the
        emissive color, which is in srgb space.
        """
        return self.uniform_buffer.data["emissive_intensity"]

    @emissive_intensity.setter
    def emissive_intensity(self, value):
        self.uniform_buffer.data["emissive_intensity"] = value
        self.uniform_buffer.update_full()

    @property
    def metalness(self):
        """How much the material looks like a metal. Non-metallic materials
        such as wood or stone use 0.0, metal use 1.0, with nothing
        (usually) in between. Default is 0.0. A value between 0.0 and
        1.0 could be used for a rusty metal look. If metalness_map is
        also provided, both values are multiplied.
        """
        return float(self.uniform_buffer.data["metalness"])

    @metalness.setter
    def metalness(self, value):
        self.uniform_buffer.data["metalness"] = value
        self.uniform_buffer.update_full()

    @property
    def metalness_map(self):
        """The blue channel of this texture is used to alter the metalness of the material."""
        return self._store.metalness_map

    @metalness_map.setter
    def metalness_map(self, map):
        assert map is None or isinstance(map, Texture)
        self._store.metalness_map = map

    @property
    def roughness(self):
        """How rough the material is. 0.0 means a smooth mirror
        reflection, 1.0 means fully diffuse. Default is 1.0.
        If roughness_map is also provided, both values are multiplied.
        """
        return float(self.uniform_buffer.data["roughness"])

    @roughness.setter
    def roughness(self, value):
        self.uniform_buffer.data["roughness"] = value
        self.uniform_buffer.update_full()

    @property
    def roughness_map(self):
        """The green channel of this texture is used to alter the roughness of the material."""
        return self._store.roughness_map

    @roughness_map.setter
    def roughness_map(self, map):
        assert map is None or isinstance(map, Texture)
        self._store.roughness_map = map

    @property
    def normal_scale(self):
        """How much the normal map affects the material. This 2-tuple
        is multiplied with the normal_map's xy components (z is
        unaffected). Typical ranges are 0-1. Default is (1,1).
        """
        return tuple(self.uniform_buffer.data["normal_scale"])

    @normal_scale.setter
    def normal_scale(self, value):
        self.uniform_buffer.data["normal_scale"] = value
        self.uniform_buffer.update_full()

    @property
    def normal_map(self):
        """The texture to create a normal map. Affects the surface
        normal for each pixel fragment and change the way the color is
        lit. Normal maps do not change the actual shape of the surface,
        only the lighting.
        """
        return self._store.normal_map

    @normal_map.setter
    def normal_map(self, map):
        assert map is None or isinstance(map, Texture)
        self._store.normal_map = map

    @property
    def env_map(self):
        """A texture that provides the environment map (in srgb colorspace). Default None.

        This makes the surroundings of the object be reflected on its
        surface. The given texture should have its ``generate_mipmaps`` set to
        True. Otherwise the roughness has no effect (as if its always zero).
        """
        # Note: to obtain a “physically correct” result, an advanced
        # mipmap technique is needed: PMREM (Prefiltered Mipmap Radiance
        # Environment Maps). We could (I think) add this technique in addition
        # to our normal mipmapping.
        return self._env_map

    @env_map.setter
    def env_map(self, map):
        assert map is None or isinstance(map, Texture)
        self._env_map = map
        if map is None:
            self.uniform_buffer.data["env_map_max_mip_level"] = 0
        else:
            if not map.generate_mipmaps:
                logger.warning(
                    "The env_map texture must have generate_mipmaps=True in order for roughness to work."
                )
            width, height, _ = map.size
            max_level = math.floor(math.log2(max(width, height))) + 1
            self.uniform_buffer.data["env_map_max_mip_level"] = float(max_level)
        self.uniform_buffer.update_full()

    @property
    def env_map_intensity(self):
        """Scales the effect of the environment map by multiplying its color.
        Note that this scaling occurs in the physical color space.
        """
        return float(self.uniform_buffer.data["env_map_intensity"])

    @env_map_intensity.setter
    def env_map_intensity(self, value):
        self.uniform_buffer.data["env_map_intensity"] = value
        self.uniform_buffer.update_full()
