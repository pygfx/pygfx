from ._base import Material
from ..utils import Color
from ..linalg import Vector3


class BackgroundMaterial(Material):
    """A background material that draws the background is a uniform color
    or in a gradient. The positional arguments are passed to ``set_colors()``.
    """

    uniform_type = dict(
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
        the botton and top. If four colors are given, it will be used for the
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
        self.uniform_buffer.update_range(0, 1)

    @property
    def color_bottom_right(self):
        """The color in the bottom right corner."""
        return Color(self.uniform_buffer.data["color_bottom_right"])

    @color_bottom_right.setter
    def color_bottom_right(self, color):
        self.uniform_buffer.data["color_bottom_right"] = Color(color)
        self.uniform_buffer.update_range(0, 1)

    @property
    def color_top_left(self):
        """The color in the top left corner."""
        return Color(self.uniform_buffer.data["color_top_left"])

    @color_top_left.setter
    def color_top_left(self, color):
        self.uniform_buffer.data["color_top_left"] = Color(color)
        self.uniform_buffer.update_range(0, 1)

    @property
    def color_top_right(self):
        """The color in the top right corner."""
        return Color(self.uniform_buffer.data["color_top_right"])

    @color_top_right.setter
    def color_top_right(self, color):
        self.uniform_buffer.data["color_top_right"] = Color(color)
        self.uniform_buffer.update_range(0, 1)


class BackgroundImageMaterial(BackgroundMaterial):
    """A background material that displays an image. If map is a 2D
    texture view, it is used as a static background. If it is a cube
    texture view, (on a NxMx6 texture) it is used as a skybox.
    """

    def __init__(self, map=None, **kwargs):
        super().__init__(**kwargs)
        self.map = map

    @property
    def map(self):
        """The texture map specifying the background image"""
        return self._map

    @map.setter
    def map(self, map):
        self._map = map


class BackgroundSkyboxMaterial(BackgroundImageMaterial):
    """A cube image background, resulting in a skybox.
    Use the up property to orient the skybox.
    """

    uniform_type = dict(
        tex_index="4xi4",
        yscale="f4",
    )

    def __init__(self, map=None, up=(0, 1, 0), **kwargs):
        super().__init__(map=map, **kwargs)
        self.up = up

    @property
    def up(self):
        """A Vector3 defining what way is up. Can be set to e.g. the
        controller's up vector. The given vector is "rounded" to the
        closest vector that is fully in one dimension.
        """
        return self._up

    @up.setter
    def up(self, value):
        if isinstance(value, (tuple, list)):
            value = Vector3(*value)
        best_score = 0
        best_up = Vector3(0, 1, 0)
        best_index = (0, 1, 2)
        best_scale = -1
        for dir, index in [
            ((1, 0, 0), (1, 0, 2)),
            ((0, 1, 0), (0, 1, 2)),
            ((0, 0, 1), (0, 2, 1)),
        ]:
            for scale in [-1, 1]:
                ref = Vector3(*[scale * v for v in dir])
                score = ref.dot(value)
                if score > best_score:
                    best_score = score
                    best_up = ref
                    best_index = index
                    best_scale = scale

        self._up = best_up
        self.uniform_buffer.data["tex_index"] = best_index + (0,)  # pad to vec4
        self.uniform_buffer.data["yscale"] = best_scale
        self.uniform_buffer.update_range(0, 1)
