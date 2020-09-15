from math import tan, pi

from ._base import Camera
from ..linalg import Matrix4


class PerspectiveCamera(Camera):
    """A 3D perspective camera.

    Parameters:
        fov (float): The field of view as an angle. Higher values give
            a wide-angle lens effect. The default is 50.
        aspect (float): The desired aspect ratio, which is used to determine
            the vision pyramid's boundaries depending on the viewport size.
            Common values are 16/9 or 4/3. Default 1.
        near (float): The near clipping plane. Must be larger than zero. Default 0.1.
        far (float): The far clipping plane. Must be larger than near. Default 2000.
    """

    def __init__(self, fov=50, aspect=1, near=0.1, far=2000):
        super().__init__()
        self.fov = float(fov)
        self.aspect = float(aspect)
        self.near = float(near)
        self.far = float(far)
        assert 0 < self.near < self.far
        self.zoom = 1
        self._view_aspect = 1

        self.update_projection_matrix()

    def __repr__(self) -> str:
        return f"PerspectiveCamera({self.fov}, {self.aspect}, {self.near}, {self.far})"

    def set_viewport_size(self, width, height):
        self._view_aspect = width / height

    def update_projection_matrix(self):
        # Get the reference width / height
        size = 2 * self.near * tan(pi / 180 * 0.5 * self.fov) / self.zoom
        # Pre-apply the reference aspect ratio
        width = size * self.aspect ** 0.5
        height = size / self.aspect ** 0.5
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
        self.projection_matrix.make_perspective(
            left, right, top, bottom, self.near, self.far
        )
        self.projection_matrix.premultiply(
            Matrix4(1, 0, 0.0, 0, 0, 1, 0.0, 0, 0.0, 0.0, 0.5, 0.0, 0, 0, 0.5, 1)
        )
        self.projection_matrix_inverse.get_inverse(self.projection_matrix)
