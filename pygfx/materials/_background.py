from ..resources import Texture
from ._base import Material
from ..utils import Color


class BackgroundMaterial(Material):
    """Solid/Gradient background color.

    A background material that draws the background is a uniform color
    or in a gradient. The positional arguments are passed to ``set_colors()``.

    Parameters
    ----------
    colors : list
        A list of 0, 1, 2, or 4 colors to use in the background. If zero,
        defaults to monochrome black. If one, the background will be monochrome
        using the given color. If two, the background will be a gradient from
        bottom to top using the given colors. If four, the background will be a
        gradient with each corner set to a different color. The value at a given
        position is then based on the relative distance of that position to each
        corner.
    kwargs : Any
        Additional kwargs are passed to the base constructor
        (:class:`pygfx.materials.Material`).

    """

    uniform_type = dict(
        Material.uniform_type,
        color_bottom_left="4xf4",
        color_bottom_right="4xf4",
        color_top_left="4xf4",
        color_top_right="4xf4",
    )

    def __init__(self, *colors, **kwargs):
        super().__init__(**kwargs)
        self.set_colors(*colors)

    def set_colors(self, *colors):
        """Set the background colors. If one color is given, it will be used
        as a uniform color. If two colors are given, it will be used for
        the bottom and top. If four colors are given, it will be used for the
        four corners.
        """
        colors = [Color(c) for c in colors]
        if len(colors) == 0:
            self.color_bottom_left = (0, 0, 0, 1)
            self.color_bottom_right = (0, 0, 0, 1)
            self.color_top_left = (0, 0, 0, 1)
            self.color_top_right = (0, 0, 0, 1)
        elif len(colors) == 1:
            self.color_bottom_left = colors[0]
            self.color_bottom_right = colors[0]
            self.color_top_left = colors[0]
            self.color_top_right = colors[0]
        elif len(colors) == 2:
            self.color_bottom_left = colors[0]
            self.color_bottom_right = colors[0]
            self.color_top_left = colors[1]
            self.color_top_right = colors[1]
        elif len(colors) == 4:
            self.color_bottom_left = colors[0]
            self.color_bottom_right = colors[1]
            self.color_top_left = colors[2]
            self.color_top_right = colors[3]
        else:
            raise ValueError("Need 1, 2 or 4 colors.")

    @property
    def color_bottom_left(self):
        """The color in the bottom left corner."""
        return Color(self.uniform_buffer.data["color_bottom_left"])

    @color_bottom_left.setter
    def color_bottom_left(self, color):
        self.uniform_buffer.data["color_bottom_left"] = Color(color)
        self.uniform_buffer.update_full()

    @property
    def color_bottom_right(self):
        """The color in the bottom right corner."""
        return Color(self.uniform_buffer.data["color_bottom_right"])

    @color_bottom_right.setter
    def color_bottom_right(self, color):
        self.uniform_buffer.data["color_bottom_right"] = Color(color)
        self.uniform_buffer.update_full()

    @property
    def color_top_left(self):
        """The color in the top left corner."""
        return Color(self.uniform_buffer.data["color_top_left"])

    @color_top_left.setter
    def color_top_left(self, color):
        self.uniform_buffer.data["color_top_left"] = Color(color)
        self.uniform_buffer.update_full()

    @property
    def color_top_right(self):
        """The color in the top right corner."""
        return Color(self.uniform_buffer.data["color_top_right"])

    @color_top_right.setter
    def color_top_right(self, color):
        self.uniform_buffer.data["color_top_right"] = Color(color)
        self.uniform_buffer.update_full()


class BackgroundImageMaterial(BackgroundMaterial):
    """Image/Skybox background.

    A background material that displays an image. If map is a 2D
    texture view, it is used as a static background. If it is a cube
    texture view, (on a NxMx6 texture) it is used as a skybox.

    Parameters
    ----------
    map : Texture
        If map is a 2D texture, it is used as static background image. If map is
        a cube texture, it is used as a skybox.
    kwargs : Any
        Additional kwargs are passed to the base constructor
        (:class:`pygfx.materials.Material`).

    """

    def __init__(self, map=None, **kwargs):
        super().__init__(**kwargs)
        self.map = map

    @property
    def map(self):
        """The texture map specifying the background image"""
        return self._store.map

    @map.setter
    def map(self, map):
        assert map is None or isinstance(map, Texture)
        self._store.map = map


class BackgroundSkyboxMaterial(BackgroundImageMaterial):
    """Skybox background.

    A cube image background, resulting in a skybox.

    """

    def __init__(self, map=None, **kwargs):
        super().__init__(map=map, **kwargs)
