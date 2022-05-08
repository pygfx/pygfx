from pygfx.linalg.vector3 import Vector3
from ._base import WorldObject
from ..utils.color import Color


class Light(WorldObject):
    """A light object."""

    uniform_type = dict(
        color="4xf4",
        intensity="f4",
    )

    def __init__(self, color=(1, 1, 1, 1), intensity=1):
        super().__init__()
        self.color = color
        self.intensity = intensity  # not used for now

    @property
    def color(self):
        return Color(self.uniform_buffer.data["color"])

    @color.setter
    def color(self, color):
        color = Color(color)
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

        # not used for now
        self.distance = distance
        self.decay = decay


class DirectionalLight(Light):
    """A light that gets emitted in a specific direction."""

    uniform_type = dict(
        target="4xf4",
    )

    def __init__(self, color=(1, 1, 1, 1), intensity=1):
        super().__init__(color, intensity)
        self.target = Vector3()  # TODO should be a WorldObject

    @property
    def target(self):
        return Vector3(*self.uniform_buffer.data["target"][:3])

    @target.setter
    def target(self, target):
        self.uniform_buffer.data["target"].flat = target.to_array()
        self.uniform_buffer.update_range(0, 1)


class AmbientLight(Light):
    """This light globally illuminates all objects in the scene equally."""
