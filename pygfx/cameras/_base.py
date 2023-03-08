import numpy as np
import pylinalg as la

from ..linalg import Matrix4, Vector3
from ..objects._base import WorldObject


class Camera(WorldObject):
    """Abstract base camera.

    Camera's are world objects and be placed in the scene, but this is not required.

    The purpose of a camera is to define the viewpoint for rendering a scene.
    This viewpoint consists of its position and orientation (in the world) and
    its projection.

    In other words, it covers the projection of world coordinates to
    normalized device coordinates (NDC), by the (inverse of) the
    camera's own world matrix and the camera's projection transform.
    The former represent the camera's position, the latter is specific
    to the type of camera.

    Parameters
    ----------
    dist : float
        The view distance factor. This represents at what distance from
        the camera the objects of interest are. It is used to set the
        near and far clipping planes, and controllers can use it as an
        indication of what is being looked at.
    up : Vector3
        The vector that is considered up in the world space. Think of it
        as pointing in the opposite direction as gravity. Default (0, 1, 0).
    zoom : float
        The zoom factor. Intended to temporary focus on a particular area. Default 1.
    """

    def __init__(self, dist=1, up=(0, 1, 0), zoom=1):
        super().__init__()

        self.dist = dist or 1
        self.up = up or (0, 1, 0)
        self.zoom = zoom or 1

        self.matrix_world_inverse = Matrix4()
        self.projection_matrix = Matrix4()
        self.projection_matrix_inverse = Matrix4()

    @property
    def dist(self):
        """The view distance factor."""
        return self._dist

    @dist.setter
    def dist(self, value):
        self._dist = float(value)

    @property
    def up(self):
        """The vector that is considered up (i.e. minus gravity) in the world space."""
        return self._up

    @up.setter
    def up(self, value):
        if isinstance(value, Vector3):
            self._up = value.clone()
        elif isinstance(value, (tuple, list)) and len(value) == 3:
            self._up = Vector3(*value)
        else:
            raise TypeError(f"Invalid up vector: {value}")

    @property
    def zoom(self):
        """The camera zoom level."""
        # todo: this thing is a bit weird. On a perspective camera, zoom is best expressed as a change in vof
        return self._zoom

    @zoom.setter
    def zoom(self, value):
        self._zoom = float(value)

    def _get_near_and_far_plane(self):
        raise NotImplementedError()

    def set_view_size(self, width, height):
        # In logical pixels
        pass

    def update_matrix_world(self, *args, **kwargs):
        super().update_matrix_world(*args, **kwargs)
        self.matrix_world_inverse.get_inverse(self.matrix_world)

    def update_projection_matrix(self):
        raise NotImplementedError()

    def look_towards(self, direction=None, up=None):
        # Note: the logic here assumes that the camera does not have a parent with a transform.
        # Let's fix that once we have an easier API to get object.forward and all that.
        if up is not None:
            self.up = up
        position = self.position.to_array()
        rotation = self.rotation.to_array()
        if direction is None:
            target = position + la.quaternion_rotate((0, 0, -self.dist), rotation)
        else:
            direction = np.asarray(direction)
            target = position + direction * self.dist / np.linalg.norm(direction)
        self.look_at(tuple(target))

    def look_at(self, target, up=None):
        # todo: the up arg makes this API different from WorldObject.look_at, but it's very convenient to be able to also set the up vector.
        if up is not None:
            self.up = up
        if isinstance(target, (tuple, np.ndarray)) and len(target) == 3:
            target = Vector3(*target)
        self._look_at(target, self._up, True)
        # self.dist = self.position.distance_to(target)

    def show_object(
        self, target: WorldObject, view_dir=(-1, -1, -1), *, up=None, size_weight=2
    ):
        """Position the camera such that the given world object in is in view.

        Parameters
        ----------
        target: WorldObject
            The object to look at.
        view_dir: 3-tuple of float or Vector3
            Look at the object from this direction.
        up: 3-tuple of float or Vector3
            Also set the up vector.
        size_weight: float
            How much extra space the camera must show.
            The target's bounding sphere radius is multiplied by this weight.
            Default 2. If you know your data is square and you look at it frontally,
            you can set it to 1.5.

        Returns:
            pos: Vector3
                The world coordinate the camera is looking at.
        """

        view_dir = tuple(view_dir)
        if len(view_dir) == 1:
            raise TypeError(f"Expected view_dir to be tuple, not {view_dir[0]}")
        elif len(view_dir) != 3:
            raise ValueError("Expected view_dir to be a tuple of 3 floats.")

        if isinstance(target, WorldObject):
            bsphere = target.get_world_bounding_sphere()
            if bsphere is None:
                pos = target.get_world_position().to_array()
                bsphere = tuple(pos) + (1, )
        elif isinstance(target, (tuple, list, np.ndarray)) and len(target) == 4:
            bsphere = tuple(target)
        else:
            raise TypeError("show_object target must be a world object, or a (x, y, z, radius) tuple.")


        view_pos = bsphere[:3]
        radius = bsphere[3]
        extent = radius * size_weight

        fov = getattr(self, "fov", None)

        if fov:
            fov_rad = fov * np.pi / 180
            distance = 0.5 * extent / np.tan(0.5 * fov_rad)
        else:
            distance = extent * 1.0

        camera_pos = view_pos - la.vector_normalize(view_dir) * distance

        self.position.set(*camera_pos)
        self.dist = extent
        self.look_at(view_pos, up)

        return view_pos

    def get_state(self):
        """Get the state of the camera as a dict."""
        return {}

    def set_state(self, state):
        """Set the state of the camera from a dict obtained with ``get_state``
        from a camera of the same type.
        """
        pass


class NDCCamera(Camera):
    """A Camera operating in NDC coordinates.

    Its projection matrix is the identity transform (but its position and rotation can still be set).

    In the NDC coordinate system of WGPU (and pygfx), x and y are in
    the range -1..1, z is in the range 0..1, and (-1, -1, 0) represents
    the bottom left corner.
    """

    def update_projection_matrix(self):
        eye = 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1
        self.projection_matrix.set(*eye)
        self.projection_matrix_inverse.set(*eye)


class ScreenCoordsCamera(Camera):
    """A Camera operating in screen coordinates.

    The depth range is the same as in NDC (0 to 1).
    """

    def __init__(self):
        super().__init__()
        self._width = 1
        self._height = 1

    def set_view_size(self, width, height):
        self._width = width
        self._height = height

    def update_projection_matrix(self):
        sx, sy, sz = 2 / self._width, 2 / self._height, 1
        dx, dy, dz = -1, -1, 0
        m = sx, 0, 0, dx, 0, sy, 0, dy, 0, 0, sz, dz, 0, 0, 0, 1
        self.projection_matrix.set(*m)
        self.projection_matrix_inverse.get_inverse(self.projection_matrix)
