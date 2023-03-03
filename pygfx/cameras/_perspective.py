from math import tan, pi

from ._base import Camera
from ..linalg import Matrix4


class PerspectiveCamera(Camera):
    """A 3D perspective camera.

    Parameters
    ----------
    fov: float
        The field of view as an angle in degrees. Higher values give a
        wide-angle lens effect. Default is 50.
    aspect : float
        The desired aspect ratio, which is used to determine the vision pyramid's
        boundaries depending on the viewport size. Common values are 16/9 or 4/3. Default 1.
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

    def __init__(self, fov=50, aspect=1, dist=1, up=(0, 1, 0), *, zoom=1):
        super().__init__(dist, up, zoom)
        self.fov = fov
        self.aspect = aspect

        self.set_view_size(1, 1)
        self.update_projection_matrix()

    def __repr__(self) -> str:
        return f"PerspectiveCamera({self.fov}, {self.aspect}, {self.dist}, {self.up})"

    @property
    def fov(self):
        """The field of view (in degrees)."""
        return self._fov

    @fov.setter
    def fov(self, value):
        self._fov = float(value)

    @property
    def aspect(self):
        """The field of view (in degrees)."""
        return self._aspect

    @aspect.setter
    def aspect(self, value):
        self._aspect = float(value)

    def _get_near_and_far_plane(self):
        d = self._dist * 90 / self.fov
        return d / 1000, 1000 * d

    def get_state(self):
        return {
            "position": tuple(self.position.to_array()),
            "rotation": tuple(self.rotation.to_array()),
            "scale": tuple(self.scale.to_array()),
            "up": tuple(self.up.to_array()),
            "fov": self.fov,
            "aspect": self.aspect,
            "dist": self.dist,
            "zoom": self.zoom,
        }

    def set_state(self, state):
        # Set the more complex props
        self.position.set(*state["position"])
        self.rotation.set(*state["rotation"])
        self.scale.set(*state["scale"])
        self.up.set(*state["up"])
        # Set simple props
        for key in ("fov", "aspect", "dist", "zoom"):
            if key in state:
                setattr(self, key, state[key])

    def set_view_size(self, width, height):
        self._view_aspect = width / height

    def update_projection_matrix(self):
        # Get the reference width / height
        near, far = self._get_near_and_far_plane()
        size = 2 * near * tan(pi / 180 * 0.5 * self.fov) / self.zoom
        # Pre-apply the reference aspect ratio
        width = size * self.aspect**0.5
        height = size / self.aspect**0.5
        # Increase eihter the width or height, depending on the view size
        if self.aspect < self._view_aspect:
            width *= self._view_aspect / self.aspect
        else:
            height *= self.aspect / self._view_aspect
        # Calculate bounds
        top = +0.5 * height
        bottom = -0.5 * height
        left = -0.5 * width
        right = +0.5 * width
        # Set matrices
        # The linalg perspective projection puts xyz in the range -1..1,
        # but in the coordinate system of wgpu (and this lib) the depth
        # is expressed in 0..1, so we also correct for that.
        self.projection_matrix.make_perspective(left, right, top, bottom, near, far)
        self.projection_matrix.premultiply(
            Matrix4(1, 0, 0.0, 0, 0, 1, 0.0, 0, 0.0, 0.0, 0.5, 0.0, 0, 0, 0.5, 1)
        )
        self.projection_matrix_inverse.get_inverse(self.projection_matrix)
