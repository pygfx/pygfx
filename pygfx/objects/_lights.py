import math

import wgpu

from ._base import WorldObject
from ..utils.color import Color
from ..linalg import Matrix4, Vector3
from ..cameras import Camera
from ..resources import Buffer
from ..cameras import OrthographicCamera, PerspectiveCamera
from ..utils import array_from_shadertype


class Light(WorldObject):
    """A light object."""

    uniform_type = dict(
        color="4xf4",
        cast_shadow='i4',
        light_view_proj_matrix="4x4xf4",
        shadow_bias="f4",
    )

    def __init__(self, color=(1, 1, 1, 1), intensity=1):
        super().__init__()
        self._intensity = intensity
        self._color = color

        self.color = color
        self.intensity = intensity

        # for internal use
        self._light_shadow = None

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

    @property
    def cast_shadow(self):
        return bool(self.uniform_buffer.data["cast_shadow"])

    @intensity.setter
    def cast_shadow(self, value: bool):
        self.uniform_buffer.data["cast_shadow"] = bool(value)

    def __update_buffer_color(self):
        # artist friendly color scaling, reference threejs
        scale_factor = self._intensity * math.pi

        color = Color(self._color).multiply_scalar(scale_factor)
        self.uniform_buffer.data["color"] = color
        self.uniform_buffer.update_range(0, 1)


class PointLight(Light):
    """A light that gets emitted from a single point in all directions."""

    uniform_type = dict(
        distance="f4",
        decay="f4",
        light_view_proj_matrix="6*4x4xf4"
    )

    def __init__(self, color=(1, 1, 1, 1), intensity=1, distance=0, decay=1):
        super().__init__(color, intensity)

        self.distance = distance
        self.decay = decay

        self.shadow = PointLightShadow()

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

        self.shadow = DirectionalLightShadow()


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

        self.shadow = SpotLightShadow()

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
    """This light globally illuminates all objects in the scene equally."""

    def __init__(self, color="#111111", intensity=1):
        super().__init__(color, intensity)


# shadows

_look_target = Vector3()
_proj_screen_matrix = Matrix4()


shadow_uniform_type =dict(
    light_view_proj_matrix="4x4xf4"
)

class LightShadow:

    def __init__(self, camera: Camera) -> None:
        self.camera = camera

        # used for pcf filtering
        # self.radius = 1
        # self.map_size = [1024, 1024]

        # used for internal rendering shadow map
        self._map = None
        self._map_index = 0

        self.bias = 0

        self.matrix_buffer = Buffer(array_from_shadertype(shadow_uniform_type))
        self.matrix_buffer._wgpu_usage = wgpu.BufferUsage.UNIFORM

    # @property
    # def bias(self):
    #     return float(self.uniform_buffer.data["bias"])

    # @bias.setter
    # def bias(self, value):
    #     self.uniform_buffer.data["bias"] = value
    #     self.uniform_buffer.update_range(0, 1)

    def update_uniform_buffers(self, light: Light):
        light.uniform_buffer.data["shadow_bias"] = self.bias
        self.update_matrix(light)
        light.uniform_buffer.update_range(0, 1)


    def update_matrix(self, light: Light) -> None:
        shadow_camera = self.camera
        shadow_camera.position.set_from_matrix_position(light.matrix_world)
        _look_target.set_from_matrix_position(light.target.matrix_world)
        shadow_camera.look_at(_look_target)
        shadow_camera.update_matrix_world()

        _proj_screen_matrix.multiply_matrices(
            shadow_camera.projection_matrix, shadow_camera.matrix_world_inverse
        )

        self.matrix_buffer.data["light_view_proj_matrix"].flat = _proj_screen_matrix.elements
        self.matrix_buffer.update_range(0, 1)

        light.uniform_buffer.data[
            "light_view_proj_matrix"
        ].flat = _proj_screen_matrix.elements


class DirectionalLightShadow(LightShadow):
    def __init__(self) -> None:
        # OrthographicCamera for directional light
        super().__init__(OrthographicCamera(100, 100, -500, 500))


class SpotLightShadow(LightShadow):
    def __init__(self) -> None:
        super().__init__(PerspectiveCamera(50, 1, 0.5, 500))
        self.focus = 1

    def update_matrix(self, light):
        camera = self.camera

        fov = 180 / math.pi * 2 * light.angle * self.focus

        aspect = 1
        far = light.distance or camera.far

        if fov != camera.fov or far != camera.far:
            camera.fov = fov
            camera.aspect = aspect
            camera.far = far
            camera.update_projection_matrix()

        super().update_matrix(light)


class PointLightShadow(LightShadow):

    # uniform_type = dict(
    #     light_view_proj_matrix="6*4x4xf4",
    #     bias="f4",
    # )

    _cube_directions = [
        Vector3(1, 0, 0),
        Vector3(-1, 0, 0),
        Vector3(0, 1, 0),
        Vector3(0, -1, 0),
        Vector3(0, 0, 1),
        Vector3(0, 0, -1),
    ]

    _cube_up = [
        Vector3(0, 1, 0),
        Vector3(0, 1, 0),
        Vector3(0, 1, 0),
        Vector3(0, 1, 0),
        Vector3(0, 0, 1),
        Vector3(0, 0, -1),
    ]

    def __init__(self) -> None:
        super().__init__(PerspectiveCamera(90, 1, 0.5, 500))

        self.matrix_buffer = []

        for _ in range(6):
            buffer = Buffer(array_from_shadertype(shadow_uniform_type))
            buffer._wgpu_usage = wgpu.BufferUsage.UNIFORM
            self.matrix_buffer.append(buffer)

    def update_matrix(self, light: Light) -> None:
        camera = self.camera

        far = light.distance or camera.far

        if far != camera.far:
            camera.far = far
            camera.update_projection_matrix()

        for i in range(6):
            camera.position.set_from_matrix_position(light.matrix_world)

            _look_target.copy(camera.position)
            _look_target.add(self._cube_directions[i])

            camera.up.copy(self._cube_up[i])

            camera.look_at(_look_target)
            camera.update_matrix_world()

            _proj_screen_matrix.multiply_matrices(
                camera.projection_matrix, camera.matrix_world_inverse
            )

            light.uniform_buffer.data[f"light_view_proj_matrix"][
                i
            ].flat = _proj_screen_matrix.elements

            self.matrix_buffer[i].data[
                "light_view_proj_matrix"
            ].flat = _proj_screen_matrix.elements
            self.matrix_buffer[i].update_range(0, 1)