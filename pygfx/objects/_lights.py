import math

from pygfx.linalg.vector3 import Vector3
from ._base import WorldObject
from ..utils.color import Color


class Light(WorldObject):
    """A light object."""

    uniform_type = dict(
        color="4xf4",
    )

    def __init__(self, color=(1, 1, 1, 1), intensity=1):
        super().__init__()
        self._intensity = intensity
        self._color = color

        self.color = color
        self.intensity = intensity

        # use for internal
        # self._needs_update = False

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = Color(color)
        self.__update_buffer_color()

        # self._needs_update = True

    @property
    def intensity(self):
        return self._intensity

    @intensity.setter
    def intensity(self, value):
        self._intensity = value
        self.__update_buffer_color()

    def __update_buffer_color(self):
        # artist friendly color scaling, reference threejs
        scale_factor = self._intensity * math.pi

        color = Color(self._color).muultiply_scalar(scale_factor)
        self.uniform_buffer.data["color"] = color
        self.uniform_buffer.update_range(0, 1)


class PointLight(Light):
    """A light that gets emitted from a single point in all directions."""

    uniform_type = dict(
        distance="f4",
        decay="f4",
    )

    def __init__(self, color=(1, 1, 1, 1), intensity=1, distance=0, decay=1):
        super().__init__(color, intensity)

        self.distance = distance
        self.decay = decay

    @property
    def distance(self):
        return float(self.uniform_buffer.data["distance"])

    @distance.setter
    def distance(self, value):
        self.uniform_buffer.data["distance"] = value
        self.uniform_buffer.update_range(0, 1)

    @property
    def decay(self):
        return float(self.uniform_buffer.data["decay"])

    @decay.setter
    def decay(self, value):
        self.uniform_buffer.data["decay"] = value
        self.uniform_buffer.update_range(0, 1)


class DirectionalLight(Light):
    """A light that gets emitted in a specific direction."""

    uniform_type = dict(
        direction="4xf4",
    )

    def __init__(self, color=(1, 1, 1, 1), intensity=1, direction=Vector3(0, -1, 0)):
        super().__init__(color, intensity)
        self.direction = direction

    @property
    def direction(self):
        return Vector3(*self.uniform_buffer.data["direction"][:3])

    @direction.setter
    def direction(self, direction):
        if isinstance(direction, (list, tuple)):
            direction = Vector3().set(*direction)

        self.uniform_buffer.data["direction"].flat = direction.to_array()
        self.uniform_buffer.update_range(0, 1)


class AmbientLight(Light):
    """This light globally illuminates all objects in the scene equally."""
