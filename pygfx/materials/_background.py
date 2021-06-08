from ._base import Material
from ..utils import array_from_shadertype
from ..resources import Buffer

# todo: in ThreeJS you can simply set a CubeTexture as the scene.background
# we could do that, and the scene could use these objects automatically.
# Or just leave it like this. I kinda like it.


class BackgroundMaterial(Material):
    """A background material that draws the background is a uniform color
    or in a gradient.
    """

    uniform_type = dict(
        color_bottom_left=("float32", 4),
        color_bottom_right=("float32", 4),
        color_top_left=("float32", 4),
        color_top_right=("float32", 4),
    )

    def __init__(self, *colors):
        super().__init__()

        self.uniform_buffer = Buffer(
            array_from_shadertype(self.uniform_type), usage="UNIFORM"
        )

        self.set_color(*colors)

    def set_color(self, *colors):
        """Set the background color. If one color is given, it will be used
        as a uniform color. If two colors are given, it will be used for
        the botton and top. If four colors are given, it will be used for the
        four corners.
        """
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
        return self.uniform_buffer.data["color_bottom_left"]

    @color_bottom_left.setter
    def color_bottom_left(self, color):
        self.uniform_buffer.data["color_bottom_left"] = color
        self.uniform_buffer.update_range(0, 1)

    @property
    def color_bottom_right(self):
        """The color in the bottom right corner."""
        return self.uniform_buffer.data["color_bottom_right"]

    @color_bottom_right.setter
    def color_bottom_right(self, color):
        self.uniform_buffer.data["color_bottom_right"] = color
        self.uniform_buffer.update_range(0, 1)

    @property
    def color_top_left(self):
        """The color in the top left corner."""
        return self.uniform_buffer.data["color_top_left"]

    @color_top_left.setter
    def color_top_left(self, color):
        self.uniform_buffer.data["color_top_left"] = color
        self.uniform_buffer.update_range(0, 1)

    @property
    def color_top_right(self):
        """The color in the top right corner."""
        return self.uniform_buffer.data["color_top_right"]

    @color_top_right.setter
    def color_top_right(self, color):
        self.uniform_buffer.data["color_top_right"] = color
        self.uniform_buffer.update_range(0, 1)


class BackgroundImageMaterial(BackgroundMaterial):
    """A background material that displays an image. If map is a 2D
    texture view, it is used as a static background. If it is a cube
    texture view, (on a NxMx6 texture) it is used as a skybox.
    Use the Background object's transform to orient the image.
    """

    def __init__(self, map=None):
        super().__init__()
        self.map = map

    @property
    def map(self):
        """The texture map specifying the background image"""
        return self._map

    @map.setter
    def map(self, map):
        self._map = map
