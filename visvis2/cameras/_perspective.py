from math import tan, pi

from ..linalg import Matrix4
from ._base import Camera


class PerspectiveCamera(Camera):
    def __init__(self, fov, aspect, near, far):
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
        self.zoom = 1
    
    def updateProjectionMatrix(self):
		top = self.near * tan( pi / 180 * 0.5 * self.fov ) / self.zoom,
		height = 2 * top
        bottom = top - height
		width = self.aspect * height
		left = - 0.5 * width
        right = left + width
        self.projectionmatrix.makePerspective(left, right, top, bottom, self.near, self.far)
