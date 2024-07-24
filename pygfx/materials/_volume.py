from ..resources import Texture
from ._base import Material
from ..utils.color import Color


class VolumeBasicMaterial(Material):
    """Basic volume material.

    Parameters
    ----------
    clim : tuple
        The contrast limits to scale the data values with. Default (0, 1).
    map : Texture
        The colormap to turn the voxel values into their final color.
    interpolation : str
        The method to interpolate the image data. Either 'nearest' or 'linear'. Default 'linear'.
    map_interpolation: str
        The method to interpolate the color map. Either 'nearest' or 'linear'. Default 'linear'.
    kwargs : Any
        Additional kwargs will be passed to the :class:`material base class
        <pygfx.Material>`.

    """

    uniform_type = dict(
        Material.uniform_type,
        clim="2xf4",
    )

    def __init__(
        self,
        clim=None,
        map=None,
        interpolation="linear",
        map_interpolation="linear",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.map = map
        self.clim = clim
        # Note: the default volume interpolation is 'linear' while it's nearest
        # for images. The ability to spot the individual voxels simply results in
        # poor visual quality.
        self.interpolation = interpolation
        self.map_interpolation = map_interpolation

    @property
    def map(self):
        """The colormap to turn the voxel values into their final color.
        If not given or None, the values themselves represents the color.
        The dimensionality of the map can be 1D, 2D or 3D, but should match the
        number of channels in the volume.
        """
        return self._store.map

    @map.setter
    def map(self, map):
        assert map is None or isinstance(map, Texture)
        self._store.map = map

    @property
    def clim(self):
        """The contrast limits to scale the data values with. Default (0, 1)."""
        v1, v2 = self.uniform_buffer.data["clim"]
        return float(v1), float(v2)

    @clim.setter
    def clim(self, clim):
        # Check and store given clim
        if clim is None:
            clim = 0, 1
        clim = float(clim[0]), float(clim[1])
        # Update uniform data
        self.uniform_buffer.data["clim"] = clim
        self.uniform_buffer.update_full()

    @property
    def interpolation(self):
        """The method to interpolate the image data. Either 'nearest' or 'linear'."""
        return self._store.interpolation

    @interpolation.setter
    def interpolation(self, value):
        assert value in ("nearest", "linear")
        self._store.interpolation = value

    @property
    def map_interpolation(self):
        """The method to interpolate the colormap. Either 'nearest' or 'linear'."""
        return self._store.map_interpolation

    @map_interpolation.setter
    def map_interpolation(self, value):
        assert value in ("nearest", "linear")
        self._store.map_interpolation = value


class VolumeSliceMaterial(VolumeBasicMaterial):
    """A material for rendering a slice through a 3D texture at the surface of a mesh.
    This material is not affected by lights.
    """

    uniform_type = dict(
        VolumeBasicMaterial.uniform_type,
        plane="4xf4",
    )

    def __init__(self, plane=(0, 0, 1, 0), **kwargs):
        super().__init__(**kwargs)
        self.plane = plane

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
        self.uniform_buffer.update_full()


class VolumeRayMaterial(VolumeBasicMaterial):
    """A material for rendering volumes using raycasting."""

    # todo: define render modes as material subclasses or using a `mode` or `style` property?
    render_mode = "mip"


class VolumeMipMaterial(VolumeRayMaterial):
    """A material rendering a volume using MIP rendering."""


class VolumeMinipMaterial(VolumeRayMaterial):
    """A material rendering a volume using MinIP rendering.

    This material renders the minimum intensity along the view ray.
    """

    render_mode = "minip"


class VolumeIsoMaterial(VolumeRayMaterial):
    """A material rendering a volume using isosurface rendering.

    Parameters
    ----------
    threshold : float
        The threshold texture value at which the surface is rendered.
        The default value is 0.5.
    step_size : float
        The size of the initial ray marching step for the initial surface finding.
        Smaller values will result in more accurate surfaces but slower rendering.
        Default value is 1.0.
    substep_size : float
        The size of the raymarching step for the refined surface finding.
        Smaller values will result in more accurate surfaces but slower rendering.
        Default value is 0.1.
    emissive : Color
        The emissive color of the surface. I.e. the color that the object emits
        even when not lit by a light source. This color is added to the final
        color and unaffected by lighting. The alpha channel is ignored.
        Default value is (0, 0, 0, 1).
    shininess : int
        How shiny the specular highlight is; a higher value gives a sharper
        highlight. Default value is 30.
    """

    render_mode = "iso"
    uniform_type = dict(
        VolumeBasicMaterial.uniform_type,
        threshold="f4",
        emissive_color="4xf4",
        shininess="f4",
        step_size="f4",
        substep_size="f4",
    )

    def __init__(
        self,
        threshold: float = 0.5,
        step_size: float = 1.0,
        substep_size: float = 0.1,
        emissive: Color = "#000",
        shininess: float = 30,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.threshold = threshold
        self.emissive = emissive
        self.shininess = shininess
        self.step_size = step_size
        self.substep_size = substep_size

    @property
    def threshold(self) -> float:
        """The threshold texture value at which the surface is rendered."""
        return self.uniform_buffer.data["threshold"]

    @threshold.setter
    def threshold(self, threshold: float) -> None:
        self.uniform_buffer.data["threshold"] = float(threshold)
        self.uniform_buffer.update_full()

    @property
    def step_size(self) -> float:
        """The size of the initial ray marching step for finding the surface.

        Smaller values will result in more accurate surfaces but slower rendering.
        """
        return self.uniform_buffer.data["step_size"]

    @step_size.setter
    def step_size(self, size: float) -> None:
        self.uniform_buffer.data["step_size"] = float(size)
        self.uniform_buffer.update_full()

    @property
    def substep_size(self) -> float:
        """The size of the raymarching step for the refined surface finding.

        Smaller values will result in more accurate surfaces but slower rendering.
        """
        return self.uniform_buffer.data["substep_size"]

    @substep_size.setter
    def substep_size(self, size: float) -> None:
        self.uniform_buffer.data["substep_size"] = float(size)
        self.uniform_buffer.update_full()

    @property
    def emissive(self):
        """The emissive (light) color of the surface.
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
    def shininess(self):
        """How shiny the specular highlight is; a higher value gives a sharper highlight.
        Default is 30.
        """
        return float(self.uniform_buffer.data["shininess"])

    @shininess.setter
    def shininess(self, value):
        self.uniform_buffer.data["shininess"] = float(value)
        self.uniform_buffer.update_full()
