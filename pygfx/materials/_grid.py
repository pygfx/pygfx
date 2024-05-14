from ._base import Material
from ..utils import Color
from ..utils.enums import CoordSpace


class GridMaterial(Material):
    """A Cartesian grid.

    Parameters
    ----------
    major_step : 2-tuple | float
        The step distance between the major ticks, for the first and second dimension, respectively.
        Default (1.0, 1.0).
    minor_step : 2-tuple | float
        The step distance between the minor ticks, for the first and second dimension, respectively.
        Default (0.1, 0.1).
    axis_thickness : float
        The thickness of the axis lines. Default 0.0 (no axis lines).
    major_thickness : float
        The thickness of the major grid lines. Default 2.0.
    minor_thickness : float
        The thickness of the minor grid lines. Default 0.75.
    thickness_space : str | CoordSpace
        The coordinate space in which the thickness is expressed ('screen', 'world', 'model'). Default 'screen'.
    axis_color : str | tuple | Color
       The color of the axis lines. Default '#777'.
    major_color : str | tuple | Color
       The color of the major grid lines. Default '#777'.
    minor_color : str | tuple | Color
       The color of the minor grid lines. Default '#777'.
    inf_grid : bool
        Whether the grid is infinite. Default True.
    kwargs : Any
        Additional kwargs are passed to the base constructor
        (:class:`pygfx.materials.Material`).
    """

    uniform_type = dict(
        Material.uniform_type,
        major_step="2xf4",
        minor_step="2xf4",
        axis_thickness="f4",
        major_thickness="f4",
        minor_thickness="f4",
        axis_color="4xf4",
        major_color="4xf4",
        minor_color="4xf4",
    )

    def __init__(
        self,
        *,
        major_step=(1.0, 1.0),
        minor_step=(0.1, 0.1),
        axis_thickness=0.0,
        major_thickness=2.0,
        minor_thickness=0.75,
        thickness_space="screen",
        axis_color="#777",
        major_color="#777",
        minor_color="#777",
        inf_grid=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.major_step = major_step
        self.minor_step = minor_step
        self.axis_thickness = axis_thickness
        self.major_thickness = major_thickness
        self.minor_thickness = minor_thickness
        self.thickness_space = thickness_space
        self.axis_color = axis_color
        self.major_color = major_color
        self.minor_color = minor_color
        self.inf_grid = inf_grid

    @property
    def major_step(self):
        """The step distance between the major grid lines."""
        return tuple(float(x) for x in self.uniform_buffer.data["major_step"])

    @major_step.setter
    def major_step(self, step):
        if isinstance(step, (float, int)):
            step = max(0.0, float(step))
            step = step, step
        if isinstance(step, (tuple, list)) and len(step) == 2:
            step = max(0.0, float(step[0])), max(0.0, float(step[1]))
        else:
            raise TypeError(
                f"major_step must be tuple or float, not {step.__class__.__name__}"
            )
        self.uniform_buffer.data["major_step"] = step
        self.uniform_buffer.update_range(0, 1)

    @property
    def minor_step(self):
        """The step distance between the minor grid lines."""
        return tuple(float(x) for x in self.uniform_buffer.data["minor_step"])

    @minor_step.setter
    def minor_step(self, step):
        if isinstance(step, (float, int)):
            step = max(0.0, float(step))
            step = step, step
        if isinstance(step, (tuple, list)) and len(step) == 2:
            step = max(0.0, float(step[0])), max(0.0, float(step[1]))
        else:
            raise TypeError(
                f"minor_step must be tuple or float, not {step.__class__.__name__}"
            )
        self.uniform_buffer.data["minor_step"] = step
        self.uniform_buffer.update_range(0, 1)

    @property
    def axis_thickness(self):
        """The thickness of the axis lines."""
        return float(self.uniform_buffer.data["axis_thickness"])

    @axis_thickness.setter
    def axis_thickness(self, thickness):
        self.uniform_buffer.data["axis_thickness"] = max(0.0, float(thickness))
        self.uniform_buffer.update_range(0, 1)

    @property
    def major_thickness(self):
        """The thickness of the major grid lines."""
        return float(self.uniform_buffer.data["major_thickness"])

    @major_thickness.setter
    def major_thickness(self, thickness):
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
        elif value not in (CoordSpace.world, CoordSpace.screen):
            raise ValueError(
                f"GridMaterial.thickness_space must be a either 'screen'  or 'world', not '{value}'."
            )
        self._store.thickness_space = value

    @property
    def axis_color(self):
        """The color of the axis lines."""
        return Color(self.uniform_buffer.data["axis_color"])

    @axis_color.setter
    def axis_color(self, color):
        self.uniform_buffer.data["axis_color"] = Color(color)
        self.uniform_buffer.update_range(0, 1)

    @property
    def major_color(self):
        """The color of the major grid lines."""
        return Color(self.uniform_buffer.data["major_color"])

    @major_color.setter
    def major_color(self, color):
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

    @property
    def inf_grid(self):
        """Whether the grid is infinite, or limited to the world-object's scale."""
        return self._store.inf_grid

    @inf_grid.setter
    def inf_grid(self, value):
        self._store.inf_grid = bool(value)
