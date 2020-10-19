from ._base import Camera
from ..linalg import Matrix4


class OrthographicCamera(Camera):
    """An orthographic camera, useful for non-perspective views and
    visualizing 2D content. You may also want to set the position of
    the camera.

    Parameters:
        width (float): The (minimum) width of the view-cube. The actual view
           may be wider if the viewport is wide.
        height (float): The (minimum) height of the view-cube.
        near (float): The near clipping plane. Default -1000.
        far (float): The far clipping plane. Must be larger than near. Default +1000.
    """

    def __init__(self, width=1, height=1, near=-1000, far=1000):
        super().__init__()
        # These width and height represent the view-plane in world coordinates
        # and has little to do with the canvas/viewport size.
        self.width = float(width)
        self.height = float(height)
        self.near = float(near)
        self.far = float(far)
        assert self.near < self.far
        self.zoom = 1
        self.maintain_aspect = True
        self.set_viewport_size(1, 1)
        self.update_projection_matrix()

    def __repr__(self) -> str:
        return (
            f"OrthographicCamera({self.width}, {self.height}, {self.near}, {self.far})"
        )

    def set_viewport_size(self, width, height):
        self._view_aspect = width / height
        # The reference view plane is scaled with the zoom factor
        width = self.width / self.zoom
        height = self.height / self.zoom
        # Increase either the width or height, depending on the viewport shape
        aspect = width / height
        if not self.maintain_aspect:
            pass
        elif aspect < self._view_aspect:
            width *= self._view_aspect / aspect
        else:
            height *= aspect / self._view_aspect
        # Store the size if the visible view plane in world coordinates
        self.visible_world_size = width, height

    def update_projection_matrix(self):
        w, h = self.visible_world_size
        bottom = -0.5 * h
        top = +0.5 * h
        left = -0.5 * w
        right = +0.5 * w
        # Set matrices
        # The linalg ortho projection puts xyz in the range -1..1, but
        # in the coordinate system of wgpu (and this lib) the depth is
        # expressed in 0..1, so we also correct for that.
        self.projection_matrix.make_orthographic(
            left, right, top, bottom, self.near, self.far
        )
        self.projection_matrix.premultiply(
            Matrix4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 1)
        )
        self.projection_matrix_inverse.get_inverse(self.projection_matrix)
