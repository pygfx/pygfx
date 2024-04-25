from ._base import Material
from ..utils import Color


class GridMaterial(Material):
    """A Cartesian grid.

    Parameters
    ----------
    color : str | tuple | Color
       The color of the grid lines.
    kwargs : Any
        Additional kwargs are passed to the base constructor
        (:class:`pygfx.materials.Material`).
    """

    uniform_type = dict(
        Material.uniform_type,
        color="4xf4",
    )

    def __init__(self, color="#777", **kwargs):
        super().__init__(**kwargs)
        self.color = color

    @property
    def color(self):
        """The color of the grid lines."""
        return Color(self.uniform_buffer.data["color"])

    @color.setter
    def color(self, color):
        self.uniform_buffer.data["color"] = Color(color)
        self.uniform_buffer.update_range(0, 1)
