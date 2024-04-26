from ._base import Material
from ..utils import Color
from ..utils.enums import CoordSpace


class GridMaterial(Material):
    """A Cartesian grid.

    Parameters
    ----------
    thickness : float
        The thickness of the major grid lines. Default 2.0.
    minor_thickness : float
        The thickness of the minor grid lines. Default 0.75.
    thickness_space : str | CoordSpace
        The coordinate space in which the thickness is expressed ('screen', 'world', 'model'). Default 'screen'.
    color : str | tuple | Color
       The color of the major grid lines. Default '#777'.
    minor_color : str | tuple | Color
       The color of the minor grid lines. Default '#777'.
    kwargs : Any
        Additional kwargs are passed to the base constructor
        (:class:`pygfx.materials.Material`).
    """

    uniform_type = dict(
        Material.uniform_type,
        major_thickness="f4",
        minor_thickness="f4",
        major_color="4xf4",
        minor_color="4xf4",
    )

    def __init__(
        self,
        thickness=2.0,
        minor_thickness=0.75,
        thickness_space="screen",
        color="#777",
        minor_color="#777",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.thickness = thickness
        self.minor_thickness = minor_thickness
        self.thickness_space = thickness_space
        self.color = color
        self.minor_color = minor_color

    @property
    def thickness(self):
        """The thickness of the major grid lines."""
        return float(self.uniform_buffer.data["major_thickness"])

    @thickness.setter
    def thickness(self, thickness):
        self.uniform_buffer.data["major_thickness"] = max(0.0, float(thickness))
        self.uniform_buffer.update_range(0, 1)

    @property
    def minor_thickness(self):
        """The thickness of the minor grid lines."""
        return float(self.uniform_buffer.data["minor_thickness"])

    @minor_thickness.setter
    def minor_thickness(self, thickness):
        self.uniform_buffer.data["minor_thickness"] = max(0.0, float(thickness))
        self.uniform_buffer.update_range(0, 1)

    @property
    def thickness_space(self):
        """The coordinate space in which the thicknesses are expressed.

        See :obj:`pygfx.utils.enums.CoordSpace`:
        """
        return self._store.thickness_space

    @thickness_space.setter
    def thickness_space(self, value):
        value = value or "screen"
        if value not in CoordSpace:
            raise ValueError(
                f"GridMaterial.thickness_space must be a string in {CoordSpace}, not {repr(value)}"
            )
        self._store.thickness_space = value
        # todo: I thick we must forbid 'model' space?

    @property
    def color(self):
        """The color of the major grid lines."""
        return Color(self.uniform_buffer.data["major_color"])

    @color.setter
    def color(self, color):
        self.uniform_buffer.data["major_color"] = Color(color)
        self.uniform_buffer.update_range(0, 1)

    @property
    def minor_color(self):
        """The color of the minor grid lines."""
        return Color(self.uniform_buffer.data["minor_color"])

    @minor_color.setter
    def minor_color(self, color):
        self.uniform_buffer.data["minor_color"] = Color(color)
        self.uniform_buffer.update_range(0, 1)
