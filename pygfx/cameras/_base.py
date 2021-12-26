from ..linalg import Matrix4, Vector3
from ..objects._base import WorldObject


class Camera(WorldObject):
    """Abstract base camera.

    The purpose of a camera is to define the viewpoint for rendering a scene.
    This viewpoint consists of its position (in the world) and its projection.

    In other words, it covers the projection of world coordinates to
    normalized device coordinates (NDC), by the (inverse of) the
    camera's own world matrix and the camera's projection transform.
    The former represent the camera's position, the latter is specific
    to the type of camera.

    Note that we follow the NDC coordinate system of WGPU, where
    x and y are in the range 0..1, z is in the range 0..1, and (-1, -1, 0)
    represents the bottom left corner.

    """

    def __init__(self):
        super().__init__()

        self.matrix_world_inverse = Matrix4()
        self.projection_matrix = Matrix4()
        self.projection_matrix_inverse = Matrix4()

    def set_view_size(self, width, height):
        # In logical pixels
        pass

    def update_matrix_world(self, *args, **kwargs):
        super().update_matrix_world(*args, **kwargs)
        self.matrix_world_inverse.get_inverse(self.matrix_world)

    def update_projection_matrix(self):
        raise NotImplementedError()

    def show_object(
        self, target: WorldObject, view_dir=(-1, -1, -1), distance_weight=2
    ):
        """Utility function to position and rotate the camera to ensure
        a particular world object is in view.

        Parameters:
            target: WorldObject
                The object to look at
            view_dir: 3-tuple of float
                Look at the object in this direction
            distance_weight: float
                The camera distance to the object's world position is
                its bounding sphere radius multiplied by this weight

        Returns:
            pos: Vector3
                The world coordinate the camera is looking at
        """
        bsphere = target.get_world_bounding_sphere()
        if bsphere is not None:
            pos, distance = Vector3(*bsphere[:3]), bsphere[3]
        else:
            pos = target.get_world_position()
            # whatever it is has no bounding sphere, so we just pick an
            # arbitrary distance
            distance = 100
        distance *= distance_weight
        self.position.copy(pos).add_scaled_vector(
            Vector3(*view_dir).normalize().negate(), distance
        )
        self.look_at(pos)
        return pos

    @property
    def flips_winding(self):
        """Get whether the camera flips any dimensions causing the
        winding of faces to be flipped. Note that if an even number of
        dimensions are flipped, the winding is not affected.
        """
        return False


class NDCCamera(Camera):
    """A Camera operating in NDC coordinates: its projection matrix
    is the identity transform (but its matrix_world can still be set).

    In the NDC coordinate system of WGPU (and pygfx), x and y are in
    the range 0..1, z is in the range 0..1, and (-1, -1, 0) represents
    the bottom left corner."""

    def update_projection_matrix(self):
        eye = 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1
        self.projection_matrix.set(*eye)
        self.projection_matrix_inverse.set(*eye)


class ScreenCoordsCamera(Camera):
    """A Camera operating in screen coordinates. The depth range is the same
    as in NDC (0 to 1).
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
