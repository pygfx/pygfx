from ._base import Camera


class OrthographicCamera(Camera):
    """ An orthographic camera, useful for non-perspective views and
    visualizing 2D content.

    Parameters:
        width (float): The (minimum) width of the view-cube. The actual view
           may be wider if the viewport is wide.
        height (float): The (minimum) height of the view-cube.
        near (float): The near clipping plane. Default -1000.
        far (float): The far clipping plane. Default +1000.
    """

    def __init__(self, width=1, height=1, near=-1000, far=1000):
        super().__init__()
        self.width = float(width)
        self.height = float(height)
        self.near = float(near)
        self.far = float(far)
        self.zoom = 1
        self._maintain_aspect = True
        self._view_aspect = 1
        self.update_projection_matrix()

    def __repr__(self) -> str:
        return (
            f"OrthographicCamera({self.width}, {self.height}, {self.near}, {self.far})"
        )

    def set_viewport_size(self, width, height):
        self._view_aspect = width / height

    def update_projection_matrix(self):
        # Get the reference width / height
        width = self.width / self.zoom
        height = self.height / self.zoom
        # Increase eihter the width or height, depending on the view size
        aspect = width / height
        if aspect < self._view_aspect:
            width *= self._view_aspect / aspect
        else:
            height *= aspect / self._view_aspect
        # Calculate bounds
        top = -0.5 * height
        bottom = +0.5 * height
        left = -0.5 * width
        right = +0.5 * width
        # Set matrices
        self.projection_matrix.make_orthographic(
            left, right, top, bottom, self.near, self.far
        )
        self.projection_matrix_inverse.get_inverse(self.projection_matrix)
