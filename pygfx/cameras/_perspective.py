from math import tan, pi

from ._base import Camera


class PerspectiveCamera(Camera):
    def __init__(self, fov, aspect, near, far):
        super().__init__()
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
        self.zoom = 1

        self.update_projection_matrix()

    def __repr__(self) -> str:
        return f"PerspectiveCamera({self.fov}, {self.aspect}, {self.near}, {self.far})"

    def set_viewport_size(self, width, height):
        self.aspect = width / height

    def update_projection_matrix(self):
        top = self.near * tan(pi / 180 * 0.5 * self.fov) / self.zoom
        height = 2 * top
        bottom = top - height
        width = self.aspect * height
        left = -0.5 * width
        right = left + width
        self.projection_matrix.make_perspective(
            left, right, top, bottom, self.near, self.far
        )
        self.projection_matrix_inverse.get_inverse(self.projection_matrix)
