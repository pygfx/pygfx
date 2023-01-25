from ._base import Material
from ..resources import TextureView
from ..utils import unpack_bitfield, Color


class LineMaterial(Material):
    """Basic line material.

    Parameters
    ----------
    color : Color
        The uniform color of the line. Ignored if ``vertex_colors`` is True.
    thickness : float
        The line thickness expressed in logical pixels.
    vertex_colors : bool
        Whether to use the vertex colors provided in the geometry to color the
        line. If True, color will be ignored.
    map : TextureView
        The texture map specifying the color for each texture coordinate. The
        dimensionality of the map can be 1D, 2D or 3D, but should match the
        number of columns in the geometry's texcoords.
    aa : bool
        Whether or not the line should be anti-aliased. Aliasing gives prettier
        results, but may affect performance for very large datasets. Default
        True.
    kwargs : Any
        Additional kwargs will be passed to the :class:`material base class
        <pygfx.Material>`.

    """

    uniform_type = dict(
        color="4xf4",
        thickness="f4",
    )

    def __init__(
        self,
        color=(1, 1, 1, 1),
        thickness=2.0,
        vertex_colors=False,
        map=None,
        aa=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.color = color
        self.aa = aa
        self.map = map
        self.thickness = thickness
        self._vertex_colors = bool(vertex_colors)

    def _wgpu_get_pick_info(self, pick_value):
        # This should match with the shader
        values = unpack_bitfield(pick_value, wobject_id=20, index=26, coord=18)
        return {
            "vertex_index": values["index"],
            "segment_coord": (values["coord"] - 100000) / 100000.0,
        }

    @property
    def color(self):
        """The uniform color of the line."""
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
    def aa(self):
        """Whether or not the line should be anti-aliased. Aliasing
        gives prettier results, but may affect performance for very large
        datasets. Default True.
        """
        return self._store.aa

    @aa.setter
    def aa(self, aa):
        self._store.aa = bool(aa)

    @property
    def vertex_colors(self):
        """Whether to use the vertex colors provided in the geometry."""
        return self._vertex_colors

    @vertex_colors.setter
    def vertex_colors(self, value):
        value = bool(value)
        if value != self._vertex_colors:
            self._vertex_colors = value

    @property
    def thickness(self):
        """The line thickness expressed in logical pixels."""
        return float(self.uniform_buffer.data["thickness"])

    @thickness.setter
    def thickness(self, thickness):
        self.uniform_buffer.data["thickness"] = thickness
        self.uniform_buffer.update_range(0, 1)

    @property
    def map(self):
        """The texture map specifying the color for each texture coordinate.
        The dimensionality of the map can be 1D, 2D or 3D, but should match the
        number of columns in the geometry's texcoords.
        """
        return self._map

    @map.setter
    def map(self, map):
        assert map is None or isinstance(map, TextureView)
        self._map = map


class LineThinMaterial(LineMaterial):
    """Thin line material.

    A simple line, drawn with line_strip primitives that has a thickness
    of one physical pixel (the thickness property is ignored).

    While you probably don't want to use this property in your application (its
    width is inconsistent and looks *very* thin on HiDPI monitors), it can be
    useful for debugging as it is more performant than other line materials.

    """


class LineThinSegmentMaterial(LineMaterial):
    """Thin line segment material.

    Simple line segments, drawn with line primitives that has a thickness
    of one physical pixel (the thickness property is ignored).

    While you probably don't want to use this property in your application (its
    width is inconsistent and looks *very* thin on HiDPI monitors), it can be
    useful for debugging as it is more performant than other line materials.

    """


class LineSegmentMaterial(LineMaterial):
    """Line segment material.

    A material that renders line segments between each two subsequent points."""


class LineArrowMaterial(LineSegmentMaterial):
    """Arrow (vector) line material.

    A material that renders line segments that look like little vectors."""
