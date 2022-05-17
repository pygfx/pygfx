import math

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

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = Color(color)
        self.__update_buffer_color()

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

    def __init__(self, color=(1, 1, 1, 1), intensity=1):
        super().__init__(color, intensity)
        # self.direction = direction
        self.target = WorldObject()
        self.position.set(0, 1, 0)  # default direction


class SpotLight(Light):
    """This light gets emitted from a single point in one direction,
    along a cone that increases in size the further from the light it gets.
    """

    uniform_type = dict(
        direction="4xf4",
        distance="f4",
        cone_cos="f4",
        penumbra_cos="f4",
        decay="f4",
    )

    def __init__(
        self,
        color=(1, 1, 1, 1),
        intensity=1,
        distance=0,
        angle=math.pi / 2,
        penumbra=0,
        decay=0,
    ):
        super().__init__(color, intensity)

        self.distance = distance
        self._angle = angle
        self._penumbra = penumbra
        self.decay = decay
        self.target = WorldObject()

        self.angle = angle
        self.penumbra = penumbra

    @property
    def distance(self):
        """Maximum range of the light. Default is 0 (no limit)."""
        return float(self.uniform_buffer.data["distance"])

    @distance.setter
    def distance(self, value):
        self.uniform_buffer.data["distance"] = value
        self.uniform_buffer.update_range(0, 1)

    @property
    def angle(self):
        """Maximum angle of light dispersion from its direction whose upper bound is Math.PI/2."""
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value
        cone_cos = math.cos(self._angle)
        self.uniform_buffer.data["cone_cos"] = cone_cos
        penumbra_cos = math.cos(self.angle * (1 - self.penumbra))
        self.uniform_buffer.data["penumbra_cos"] = penumbra_cos
        self.uniform_buffer.update_range(0, 1)

    @property
    def penumbra(self):
        """Percent of the spotlight cone that is attenuated due to penumbra.
        Takes values between zero and 1. Default is zero."""
        return self._penumbra

    @penumbra.setter
    def penumbra(self, value):
        self._penumbra = value
        penumbra_cos = math.cos(self.angle * (1 - self.penumbra))
        self.uniform_buffer.data["penumbra_cos"] = penumbra_cos
        self.uniform_buffer.update_range(0, 1)

    @property
    def decay(self):
        """The amount the light dims along the distance of the light."""
        return float(self.uniform_buffer.data["decay"])

    @decay.setter
    def decay(self, value):
        self.uniform_buffer.data["decay"] = value
        self.uniform_buffer.update_range(0, 1)


class AmbientLight(Light):
    def __init__(self, color="#111111", intensity=1):
        super().__init__(color, intensity)

    """This light globally illuminates all objects in the scene equally."""
