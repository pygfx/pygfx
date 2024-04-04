from ..resources import Texture
from ._base import Material
from ..utils import Color, unpack_bitfield


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

    def _wgpu_get_pick_info(self, pick_value):
        # This should match with the shader
        values = unpack_bitfield(pick_value, wobject_id=20, x=22, y=22)
        # TODO: choose a convention to return to the user
        # Presently:
        # Bottom left: (0, 0)
        # Bottom right: (1, 0)
        # Top left: (0, 1)
        # Top right: (1, 1)
        # More visually:
        #     (0, 1)    (1, 1)
        #     (0, 0),   (1, 0)
        x = values["x"] / 4194303
        y = values["y"] / 4194303
        return {
            "coord": (x, y),
        }


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
        return self._map

    @map.setter
    def map(self, map):
        assert map is None or isinstance(map, Texture)
        self._map = map

    def _wgpu_get_pick_info(self, pick_value):
        tex = self.map

        # This should match with the shader
        if tex.size[2] == 1:
            # 2D
            values = unpack_bitfield(pick_value, wobject_id=20, x=22, y=22)
            # +- 0.5???
            # While the image typically uses a (-0.5) offset when picking
            # pixels in the current implementation of the background image,
            # where the image extends from edge to edge regardless of aspect
            # ratio, the offset does not seems like it is entirely meaningful.
            # It may be more appropriate when we decide to maintain the aspect
            # ratio of the texture

            # Presently these match with the spirit of the BackgroundMaterial
            # above, but instead of "1" we return "max"
            # Bottom left: (0, 0)
            # Bottom right: (1, 0)
            # Top left: (0, 1)
            # Top right: (1, 1)
            # More visually:
            #     (0, 1)    (1, 1)
            #     (0, 0),   (1, 0)
            x = values["x"] / 4194303 * tex.size[0]
            y = values["y"] / 4194303 * tex.size[1]
            return {
                "coord": (x, y),
            }
        else:  # tex.size[2] == 6
            # Cube / Skybox
            values = unpack_bitfield(pick_value, wobject_id=20, x=14, y=14, z=14)
            # unit vector pointing in the way of the pick event
            direction = (
                (values["x"] - 8192) / 8191,
                (values["y"] - 8192) / 8191,
                (values["z"] - 8192) / 8191
            )
            print(direction)
            # Do we want the face index?
            # size = tex.size
            # face_index, u, v = cube_map_coord(direction)
            return {
                "coord": direction,
            }

import numpy as np

# From chatgpt... i cheated....
def cube_map_coord(direction):
    x, y, z = direction
    abs_x, abs_y, abs_z = abs(x), abs(y), abs(z)

    is_x_positive = x > 0
    is_y_positive = y > 0
    is_z_positive = z > 0

    max_axis, uc, vc = 0, 0, 0

    # Determine which face of the cubemap we're on.
    if abs_x >= abs_y and abs_x >= abs_z:
        # Left or right face.
        max_axis = abs_x
        uc = -z if is_x_positive else z
        vc = -y
        face_index = 0 if is_x_positive else 1
    elif abs_y >= abs_x and abs_y >= abs_z:
        # Top or bottom face.
        max_axis = abs_y
        uc = x
        vc = z if is_y_positive else -z
        face_index = 2 if is_y_positive else 3
    elif abs_z >= abs_x and abs_z >= abs_y:
        # Front or back face.
        max_axis = abs_z
        uc = x
        vc = -y
        face_index = 4 if is_z_positive else 5

    # Convert range from -1 to 1 to 0 to 1
    u = 0.5 * (uc / max_axis + 1.0)
    v = 0.5 * (vc / max_axis + 1.0)

    return face_index, u, v

class BackgroundSkyboxMaterial(BackgroundImageMaterial):
    """Skybox background.

    A cube image background, resulting in a skybox.

    """
