import math

import wgpu

from ._base import WorldObject
from ..utils.color import Color
from ..linalg import Matrix4, Vector3
from ..cameras import Camera
from ..resources import Buffer
from ..cameras import OrthographicCamera, PerspectiveCamera
from ..utils import array_from_shadertype


def get_pos_from_camera_parent_or_target(light):
    if isinstance(light.parent, Camera):
        p = light.position.clone().add(Vector3(0, 0, -1))
        return p.apply_matrix4(light.parent.matrix_world)
    else:
        return light.target.get_world_position()


class Light(WorldObject):
    """The base light object.

    Parameters:
        color (Color): The base color of the light.
        intensity (float): The light intensity.
        cast_shadow (bool): Whether the light can cast shadows. Default False.
    """

    # Note that for lights and shadows, the uniform data is stored on the environment.
    # We can use the uniform_buffer as usual though. We'll just copy it over.

    uniform_type = dict(
        color="4xf4",
        intensity="f4",
        cast_shadow="i4",
        light_view_proj_matrix="4x4xf4",
        shadow_bias="f4",
    )

    def __init__(self, color="#ffffff", intensity=1, *, cast_shadow=False, **kwargs):
        super().__init__(**kwargs)
        self.color = color
        self.intensity = intensity
        self.cast_shadow = cast_shadow

        # for internal use
        self._shadow = None

    def _gfx_update_uniform_buffer(self):
        # _gfx prefix means that its not a public method, but other parts in pygfx can use it.
        pass

    @property
    def shadow(self):
        """The shadow object for this light."""
        return self._shadow

    @property
    def color(self):
        """The color of the light, in the srgb colorspace."""
        return Color(self.uniform_buffer.data["color"])

    @color.setter
    def color(self, color):
        self.uniform_buffer.data["color"] = Color(color)
        self.uniform_buffer.update_range(0, 1)

    @property
    def intensity(self):
        """The light intensity as a float, default 1.0. The units of
        intensity depend on the type of light. For point and spot lights
        it represents the luminous intensity of the light measured in
        candela (cd).

        The intensity scales the color in the physical colorspace, as
        if scaling the number of photons. Note that an intensity of 0.5
        is not equivalent to halving the color value. This is because
        the srgb color is perceptually linear, while intensity is
        physically linear. Values over 1.0 make perfect sense - it's
        just a brighter light.
        """
        return float(self.uniform_buffer.data["intensity"])

    @intensity.setter
    def intensity(self, value):
        self.uniform_buffer.data["intensity"] = float(value)
        self.uniform_buffer.update_range(0, 1)

    @property
    def cast_shadow(self):
        """Whether or not this light will casts shadows on objects.
        Note that shadows are only cast on objects that have receive_shadow
        set to True.
        """
        return bool(self.uniform_buffer.data["cast_shadow"])

    @cast_shadow.setter
    def cast_shadow(self, value: bool):
        self.uniform_buffer.data["cast_shadow"] = bool(value)


class AmbientLight(Light):
    """A light that globally illuminates all objects in the scene equally.

    Parameters:
        color (Color): The base color of the light. Default white.
        intensity (float): The light intensity. Default 0.2, corresponding to a dimly lit scene.
    """

    def __init__(self, color="#ffffff", intensity=0.2):
        super().__init__(color, intensity)


class PointLight(Light):
    """A light that gets emitted from a single point in all directions.

    Parameters:
        color (Color): The base color of the light. Default white.
        intensity (float): The light intensity. Default 3, resulting in a well lit scene.
        cast_shadow (bool): Whether the light can cast shadows. Default False.
        distance (float): The maximum distance that the light shines.
            Default is 0 which means it shines infinitely far.
        decay (float): The amount the light dims along the distance of the light.
            Default 0, no decay. A decay of 2 is physically correct. Note that you
            probably need to increase the intensity (a lot) when setting this to nonzero.
    """

    uniform_type = dict(distance="f4", decay="f4", light_view_proj_matrix="6*4x4xf4")

    def __init__(
        self,
        color="#ffffff",
        intensity=3,
        *,
        cast_shadow=False,
        distance=0,
        decay=0,
        **kwargs,
    ):
        super().__init__(color, intensity, cast_shadow=cast_shadow, **kwargs)
        self.distance = distance
        self.decay = decay
        self._shadow = PointLightShadow()

    @property
    def power(self):
        """The light's power. I.e. the luminous power of the light measured in lumens (lm).
        Changing the power will also change the light's intensity.
        """
        # compute the light's luminous power (in lumens) from its intensity (in candela)
        # for an isotropic light source, luminous power (lm) = 4 π luminous intensity (cd)
        return self.intensity * 4 * math.PI

    @power.setter
    def power(self, power):
        # set the light's intensity (in candela) from the desired luminous power (in lumens)
        self.intensity = power / (4 * math.PI)

    @property
    def distance(self):
        """When distance is zero, light will attenuate according to
        inverse-square law to infinite distance. When distance is
        non-zero, light will attenuate the same way until near the
        distance cutoff, where it will then attenuate quickly and
        smoothly to 0. Inherently, cutoffs are not physically correct.
        """
        return float(self.uniform_buffer.data["distance"])

    @distance.setter
    def distance(self, value):
        self.uniform_buffer.data["distance"] = value
        self.uniform_buffer.update_range(0, 1)

    @property
    def decay(self):
        """The amount the light dims along the distance of the light.
        A decay of 2 leads to physically realistic light falloff.
        Default is 0, which means the light does not decay.
        """
        return float(self.uniform_buffer.data["decay"])

    @decay.setter
    def decay(self, value):
        self.uniform_buffer.data["decay"] = value
        self.uniform_buffer.update_range(0, 1)


class DirectionalLight(Light):
    """A light that gets emitted in a direction, specified by its
    position and a target. If attached to a camera, the camera view
    direction is followed.

    Parameters:
        color (Color): The base color of the light. Default white.
        intensity (float): The light intensity. Default 3, resulting in a well lit scene.
        cast_shadow (bool): Whether the light can cast shadows. Default False.
        target (WorldObject): The object to direct the light at.
    """

    uniform_type = dict(
        direction="4xf4",
    )

    def __init__(
        self, color="#ffffff", intensity=3, *, cast_shadow=False, target=None, **kwargs
    ):
        super().__init__(color, intensity, cast_shadow=cast_shadow, **kwargs)
        self.target = target or WorldObject()
        self._shadow = DirectionalLightShadow()
        self._gfx_distance_to_target = 0

    @property
    def target(self):
        """The object to direct the light at. The light direction is
        from its position to its target.

        However, if the light's parent is a camera, it follows the
        camera direction instead (thus ignoring the target).
        """
        return self._target

    @target.setter
    def target(self, target):
        assert isinstance(target, WorldObject)
        self._target = target

    def _gfx_update_uniform_buffer(self):
        pos1 = self.get_world_position()
        pos2 = get_pos_from_camera_parent_or_target(self)
        origin_to_target = Vector3().sub_vectors(pos2, pos1)
        self._gfx_distance_to_target = origin_to_target.length()
        if self._gfx_distance_to_target > 0:
            direction = origin_to_target.normalize()
        else:
            direction = Vector3(0, 0, -1)  # ill-defined direction -> look neg z-axis
        self.uniform_buffer.data["direction"].flat = direction.to_array()
        self.look_at(pos2)


class SpotLight(Light):
    """A light that gets emitted from a single point in one direction,
    along a cone that increases in size the further from the light it
    gets. If attached to a camera, the camera view direction is followed.

    Parameters:
        color (Color): The base color of the light. Default white.
        intensity (float): The light intensity. Default 3, resulting in a well lit scene.
        cast_shadow (bool): Whether the light can cast shadows. Default False.
        distance (float): The maximum distance that the light shines.
            Default is 0 which means it shines infinitely far.
        decay (float): The amount the light dims along the distance of the light.
            Default 0, no decay. A decay of 2 is physically correct. Note that you
            probably need to increase the intensity (a lot) when setting this to nonzero.
        angle (float): The maximum extent of the spotlight, in radians. Default Math.PI/3.
        penumbra (float): Percent of the spotlight cone that is attenuated due
            to penumbra. Takes values between zero and 1. Default is zero.
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
        color="#ffffff",
        intensity=3,
        *,
        cast_shadow=False,
        distance=0,
        decay=0,
        angle=math.pi / 3,
        penumbra=0,
        **kwargs,
    ):
        super().__init__(color, intensity, cast_shadow=cast_shadow, **kwargs)

        self.distance = distance
        self._angle = angle
        self._penumbra = penumbra
        self.decay = decay
        self.target = WorldObject()

        self.angle = angle
        self.penumbra = penumbra

        self._shadow = SpotLightShadow()

    def _gfx_update_uniform_buffer(self):
        pos1 = self.get_world_position()
        pos2 = get_pos_from_camera_parent_or_target(self)
        origin_to_target = Vector3().sub_vectors(pos2, pos1)
        self._gfx_distance_to_target = origin_to_target.length()
        if self._gfx_distance_to_target > 0:
            direction = origin_to_target.normalize()
        else:
            direction = Vector3(0, 0, -1)  # ill-defined direction -> look neg z-axis
        self.uniform_buffer.data["direction"].flat = direction.to_array()
        self.look_at(pos2)

    @property
    def power(self):
        """The light's power. I.e. the luminous power of the light measured in lumens (lm).
        Changing the power will also change the light's intensity.
        """
        # compute the light's luminous power (in lumens) from its intensity (in candela)
        # by convention for a spotlight, luminous power (lm) = π * luminous intensity (cd)
        return self.intensity * math.PI

    @power.setter
    def power(self, power):
        # set the light's intensity (in candela) from the desired luminous power (in lumens)
        self.intensity = power / math.PI

    @property
    def distance(self):
        """When distance is zero, light will attenuate according to
        inverse-square law to infinite distance. When distance is
        non-zero, light will attenuate the same way until near the
        distance cutoff, where it will then attenuate quickly and
        smoothly to 0. Inherently, cutoffs are not physically correct.
        """
        return float(self.uniform_buffer.data["distance"])

    @distance.setter
    def distance(self, value):
        self.uniform_buffer.data["distance"] = value
        self.uniform_buffer.update_range(0, 1)

    @property
    def angle(self):
        """The maximum extent of the spotlight, in radians, from its
        direction. Should be no more than Math.PI/2.
        """
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
        Takes values between zero and 1.
        """
        return self._penumbra

    @penumbra.setter
    def penumbra(self, value):
        self._penumbra = value
        penumbra_cos = math.cos(self.angle * (1 - self.penumbra))
        self.uniform_buffer.data["penumbra_cos"] = penumbra_cos
        self.uniform_buffer.update_range(0, 1)

    @property
    def decay(self):
        """The amount the light dims along the distance of the light.
        A decay of 2 leads to physically realistic light falloff.
        Default is 0, which means the light does not decay.
        """
        return float(self.uniform_buffer.data["decay"])

    @decay.setter
    def decay(self, value):
        self.uniform_buffer.data["decay"] = value
        self.uniform_buffer.update_range(0, 1)


# shadows

_look_target = Vector3()
_proj_screen_matrix = Matrix4()


shadow_uniform_type = dict(light_view_proj_matrix="4x4xf4")


class LightShadow:
    """
    A shadow map for a light. This is usually created automatically by
    a light, and can be accessed through the light's shadow property.

    Parameters:
        camera: The light's view of the world. This is used to generate
            a depth map of the scene; objects behind other objects from
            the light's perspective will be in shadow.
    """

    def __init__(self, camera: Camera) -> None:
        self._camera = camera
        self._camera.maintain_aspect = False

        # TODO: 'radius' represents the shadow sampling radius,
        # which is used for PCF to blur and smooth shadow edges.
        # But it seems difficult to be used as a uniform in shader internal.
        # Changing this value will cause the shader to recompile.
        # Shadows with different radius in one scene are also difficult to handle.
        # now, it is a fixed value in shader.
        # self._radius = 1
        # self._map_size = [1024, 1024]

        self.bias = 0

        self._gfx_matrix_buffer = Buffer(array_from_shadertype(shadow_uniform_type))
        self._gfx_matrix_buffer._wgpu_usage = wgpu.BufferUsage.UNIFORM

    @property
    def camera(self):
        """The camera that defines the POV for determining the depth map of the scene."""
        return self._camera

    @property
    def bias(self):
        """Shadow map bias. Very tiny adjustments here may help reduce artifacts in shadows."""
        return self._bias

    @bias.setter
    def bias(self, value):
        self._bias = float(value)

    def _gfx_update_uniform_buffer(self, light: Light):
        light.uniform_buffer.data["shadow_bias"] = self._bias
        self._update_matrix(light)
        light.uniform_buffer.update_range(0, 1)

    def _update_matrix(self, light: Light) -> None:
        shadow_camera = self.camera
        shadow_camera.position.set_from_matrix_position(light.matrix_world)
        _look_target.copy(get_pos_from_camera_parent_or_target(light))
        shadow_camera.look_at(_look_target)
        shadow_camera.update_matrix_world()

        _proj_screen_matrix.multiply_matrices(
            shadow_camera.projection_matrix, shadow_camera.matrix_world_inverse
        )

        self._gfx_matrix_buffer.data[
            "light_view_proj_matrix"
        ].flat = _proj_screen_matrix.elements
        self._gfx_matrix_buffer.update_range(0, 1)

        light.uniform_buffer.data[
            "light_view_proj_matrix"
        ].flat = _proj_screen_matrix.elements


class DirectionalLightShadow(LightShadow):
    """A shadow for a directional light source."""

    def __init__(self) -> None:
        # OrthographicCamera for directional light
        super().__init__(OrthographicCamera(1000, 1000, -500, 500))

    def _update_matrix(self, light):
        camera = self.camera
        camera.update_projection_matrix()
        super()._update_matrix(light)


class SpotLightShadow(LightShadow):
    """A shadow for a spot light source."""

    def __init__(self) -> None:
        super().__init__(PerspectiveCamera(50, 1, 0.5, 500))
        self._focus = 1

    def _update_matrix(self, light):
        camera = self.camera

        fov = 180 / math.pi * 2 * light.angle * self._focus

        aspect = 1
        far = light.distance or camera.far

        if fov != camera.fov or far != camera.far:
            camera.fov = fov
            camera.aspect = aspect
            camera.far = far
            camera.update_projection_matrix()

        super()._update_matrix(light)


class PointLightShadow(LightShadow):
    """A shadow for a point light source."""

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

        self._gfx_matrix_buffer = []

        for _ in range(6):
            buffer = Buffer(array_from_shadertype(shadow_uniform_type))
            buffer._wgpu_usage = wgpu.BufferUsage.UNIFORM
            self._gfx_matrix_buffer.append(buffer)

    def _update_matrix(self, light: Light) -> None:
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

            self._gfx_matrix_buffer[i].data[
                "light_view_proj_matrix"
            ].flat = _proj_screen_matrix.elements
            self._gfx_matrix_buffer[i].update_range(0, 1)
