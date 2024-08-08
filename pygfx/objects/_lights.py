import math

import pylinalg as la
import numpy as np

from ._base import WorldObject
from ..utils.color import Color
from ..cameras import Camera
from ..resources import Buffer
from ..cameras import OrthographicCamera, PerspectiveCamera
from ..utils import array_from_shadertype


def get_pos_from_camera_parent_or_target(light: "Light") -> np.ndarray:
    if isinstance(light.parent, Camera):
        cam = light.parent
        transform = cam.world.matrix
        return la.vec_transform((0, 0, -1), transform)
    elif isinstance(light, SpotLight):
        return light.target.world.position
    elif isinstance(light, DirectionalLight):
        return light.target.world.position
    else:
        raise ValueError("Unknown light source.")


class Light(WorldObject):
    """Light Base Class.

    Parameters
    ----------
    color : Color
        The color of the light emitted.
    intensity : float
        The light's intensity. Its units depend on the type of light. For point
        and spot lights it represents the luminous intensity of the light
        measured in candela (cd).
    cast_shadow : bool
        If True, the light can cast shadows. Otherwise it doesn't.

    Notes
    -----
    The light's intensity scales the color in the physical colorspace, as if
    scaling the number of photons. Note that an intensity of 0.5 is not
    equivalent to halving the color value. This is because the srgb color is
    perceptually linear, while intensity is physically linear. Values over 1.0
    make perfect sense - it's just a brighter light.

    There are two booleans that control the behavior of shadow: `cast_shadow` on
    a Light, and `receive_shadow` on the illuminated object. Shadows will only
    be displayed if both are set to True.

    """

    # Note that for lights and shadows, the uniform data is stored on the environment.
    # We can use the uniform_buffer as usual though. We'll just copy it over.

    _FORWARD_IS_MINUS_Z = True

    uniform_type = dict(
        WorldObject.uniform_type,
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
        self.uniform_buffer.update_full()

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
        self.uniform_buffer.update_full()

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
    """Ambient light source.

    A light that omnidirectionally illuminates all objects in the scene equally.

    Parameters
    ----------
    color : Color
        The color of the emitted light.
    intensity : float
        The light intensity. A value of ``0.2`` corresponds to a dimly lit
        scene.

    """

    def __init__(self, color="#ffffff", intensity=0.2):
        super().__init__(color, intensity)


class PointLight(Light):
    """Radial point light source.

    A light that gets emitted from a single point in all directions.

    Parameters
    ----------
    color : Color
        The color of the emitted light.
    intensity : float
        The light intensity. A value of ``3`` corresponds to a well lit
        scene.
    cast_shadow : bool
        If True, the light can cast shadows. Otherwise it doesn't.
    distance : float
        The maximum distance at which objects are considered illuminated by the
        light. Limiting this may increase performance in large scenes. A value
        of ``0`` means that all objects are considered.
    decay : float
        The rate at which the light dims as it travels. A value of ``0`` means
        no decay. A decay of ``2`` is physically correct.

    Notes
    -----
    When setting ``decay`` to a non-zero value you likely probably have to
    increase intensity (a lot) as light decays following an inverse-square
    profile.

    """

    uniform_type = dict(
        Light.uniform_type,
        distance="f4",
        decay="f4",
        light_view_proj_matrix="6*4x4xf4",
    )

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
        return self.intensity * 4 * np.pi

    @power.setter
    def power(self, power):
        # set the light's intensity (in candela) from the desired luminous power (in lumens)
        self.intensity = power / (4 * np.pi)

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
        self.uniform_buffer.update_full()

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
        self.uniform_buffer.update_full()


class DirectionalLight(Light):
    """Directional light source.

    A light that gets emitted in a direction, specified by its position and a
    target. This is equivalent to an infinitely large softbox.

    Parameters
    ----------
    color : Color
        The color of the light emitted.
    intensity : float
        The light intensity. A value of ``3`` corresponds to a well lit
        scene.
    cast_shadow : bool
        If True, the light can cast shadows. Otherwise it doesn't.
    target : WorldObject
        The object used to determine the light's direction. The light will shine
        from it's position toward's the direction of the target except when the
        light's parent is a camera, in which case target is ignored.

    Notes
    -----
    If this light is attached to a camera it's direction will follow the camera
    view direction.

    There are two booleans that control the behavior of shadow: `cast_shadow` on
    a Light, and `receive_shadow` on the illuminated object. Shadows will only
    be displayed if both are set to True.

    """

    uniform_type = dict(
        Light.uniform_type,
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
        pos1 = self.world.position
        pos2 = get_pos_from_camera_parent_or_target(self)
        origin_to_target = pos2 - pos1
        self._gfx_distance_to_target = np.linalg.norm(origin_to_target)
        if self._gfx_distance_to_target > 0:
            direction = origin_to_target / self._gfx_distance_to_target
        else:
            direction = np.array(
                (0, 0, -1), dtype=float
            )  # ill-defined direction -> look neg z-axis
        self.uniform_buffer.data["direction"].flat = direction
        self.look_at(pos1 + direction)


class SpotLight(Light):
    """Directional point light source.

    A light that gets emitted from a single point in one direction, along a cone
    that increases in size the further from the light it gets.

    Parameters
    ----------
    color : Color
        The color of the light emitted.
    intensity : float
        The light intensity. A value of ``3`` corresponds to a well lit
        scene.
    cast_shadow : bool
        If True, the light can cast shadows. Otherwise it doesn't.
    distance : float
        The maximum distance at which objects are considered illuminated by the
        light. Limiting this may increase performance in large scenes. A value
        of ``0`` means that all objects are considered.
    decay : float
        The rate at which the light dims as it travels. A value of ``0`` means
        no decay. A decay of ``2`` is physically correct.
    angle : float
        The central angle (in rad) of the light's cone.
    penumbra : float
        Percent of the spotlight cone that is attenuated due
        to penumbra. Takes values between zero and 1. Default is zero.

    Notes
    -----
    If this light is attached to a camera it's direction will follow the camera
    view direction.

    There are two booleans that control the behavior of shadow: `cast_shadow` on
    a Light, and `receive_shadow` on the illuminated object. Shadows will only
    be displayed if both are set to True.

    When setting ``decay`` to a non-zero value you likely probably have to
    increase intensity (a lot) as light decays following an inverse-square
    profile.

    """

    uniform_type = dict(
        Light.uniform_type,
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
        pos1 = self.world.position
        pos2 = get_pos_from_camera_parent_or_target(self)
        origin_to_target = pos2 - pos1
        self._gfx_distance_to_target = np.linalg.norm(origin_to_target)
        if self._gfx_distance_to_target > 0:
            direction = origin_to_target / self._gfx_distance_to_target
        else:
            direction = np.array(
                (0, 0, -1), dtype=float
            )  # ill-defined direction -> look neg z-axis
        self.uniform_buffer.data["direction"].flat = direction
        self.look_at(pos1 + direction)

    @property
    def power(self):
        """The light's power. I.e. the luminous power of the light measured in lumens (lm).
        Changing the power will also change the light's intensity.
        """
        # compute the light's luminous power (in lumens) from its intensity (in candela)
        # by convention for a spotlight, luminous power (lm) = π * luminous intensity (cd)
        return self.intensity * np.pi

    @power.setter
    def power(self, power):
        # set the light's intensity (in candela) from the desired luminous power (in lumens)
        self.intensity = power / np.pi

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
        self.uniform_buffer.update_full()

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
        self.uniform_buffer.update_full()

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
        self.uniform_buffer.update_full()

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
        self.uniform_buffer.update_full()


# shadows
shadow_uniform_type = dict(light_view_proj_matrix="4x4xf4")


class LightShadow:
    """Shadow map utility base class.

    This is usually created automatically by a light, and can be accessed
    through the light's shadow property.

    Parameters
    ----------
    camera : Camera
        The light's view of the scene. This is used to generate a depth map of
        the scene; objects occluded by other objects from the light's
        perspective will receive shadow.

    """

    def __init__(self, camera: PerspectiveCamera) -> None:
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

        # TODO: move bias and cull_mode to Light so they can be reactive?
        self.bias = 0
        self.cull_mode = "front"
        self._gfx_matrix_buffer = Buffer(
            array_from_shadertype(shadow_uniform_type), force_contiguous=True
        )

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

    @property
    def cull_mode(self):
        """
        Shadow map cull_mode ('front', 'back', or 'none'). When shadow
        mapping open meshes, set to 'none' and increase ``bias`` value
        to avoid shadow acne.
        """
        return self._cull_mode

    @cull_mode.setter
    def cull_mode(self, value):
        value = str(value).lower()
        if value in ("front", "back", "none"):
            self._cull_mode = value
        else:
            raise ValueError(f"invalid cull_mode: '{value}'")

    def _gfx_update_uniform_buffer(self, light: Light):
        light.uniform_buffer.data["shadow_bias"] = self._bias
        self._update_matrix(light)
        light.uniform_buffer.update_full()

    def _update_matrix(self, light: Light) -> None:
        shadow_camera = self.camera
        shadow_camera.world.position = light.world.position
        target = get_pos_from_camera_parent_or_target(light)
        shadow_camera.look_at(target)

        self._gfx_matrix_buffer.data["light_view_proj_matrix"] = (
            shadow_camera.camera_matrix.T
        )
        self._gfx_matrix_buffer.update_full()

        light.uniform_buffer.data["light_view_proj_matrix"] = (
            shadow_camera.camera_matrix.T
        )


class DirectionalLightShadow(LightShadow):
    """Shadow map utility for directional lights."""

    def __init__(self) -> None:
        # OrthographicCamera for directional light
        super().__init__(OrthographicCamera(1000, 1000, depth_range=(-500, 500)))

    def _update_matrix(self, light):
        camera = self.camera
        camera.update_projection_matrix()
        super()._update_matrix(light)


class SpotLightShadow(LightShadow):
    """Shadow map utility for spot light sources."""

    def __init__(self) -> None:
        super().__init__(PerspectiveCamera(50, depth_range=(0.5, 500)))
        self._focus = 1

    def _update_matrix(self, light):
        camera = self.camera

        fov = 180 / math.pi * 2 * light.angle * self._focus

        aspect = 1
        far = (light.distance * 10) or camera.far

        if fov != camera.fov or far != camera.far:
            camera.fov = fov
            camera.aspect = aspect
            camera.depth_range = far / 1000000, far
            camera.update_projection_matrix()

        super()._update_matrix(light)


class PointLightShadow(LightShadow):
    """Shadow map utility for point light sources."""

    _cube_directions = np.array(
        [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ],
        dtype=float,
    )

    def __init__(self) -> None:
        super().__init__(PerspectiveCamera(90))

        self._gfx_matrix_buffer = []

        for _ in range(6):
            buffer = Buffer(
                array_from_shadertype(shadow_uniform_type), force_contiguous=True
            )
            self._gfx_matrix_buffer.append(buffer)

    def _update_matrix(self, light: Light) -> None:
        camera = self.camera
        camera.world.position = light.world.position
        directions = self._cube_directions + light.world.position

        far = (light.distance * 10) or camera.far

        if far != camera.far:
            camera.depth_range = far / 1000000, far
            camera.update_projection_matrix()

        for i in range(6):
            # Note: the direction may align with `up`, but we have logic in
            # `look_at` to catch and handle this special case.
            camera.look_at(directions[i])

            light.uniform_buffer.data["light_view_proj_matrix"][
                i
            ] = camera.camera_matrix.T
            self._gfx_matrix_buffer[i].data[
                "light_view_proj_matrix"
            ] = camera.camera_matrix.T
            self._gfx_matrix_buffer[i].update_full()
        light.uniform_buffer.update_full()
