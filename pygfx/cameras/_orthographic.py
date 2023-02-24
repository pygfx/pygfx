from ._base import Camera
from ..linalg import Matrix4


class OrthographicCamera(Camera):
    """An orthographic camera, useful for non-perspective views and 2D content.

    Parameters
    ----------
    width : float
        The (minimum) width of the view-cube. The actual view
        may be wider if the viewport is relatively wide.
    height : float
        The (minimum) height of the view-cube. The actual view
        may be height if the viewport is relatively heigh.
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
    maintain_aspect : bool
        Whether the aspect ration is maintained as the window size changes. Default True.

    """

    def __init__(
        self, width, height, dist=1, up=(0, 1, 0), *, zoom=1, maintain_aspect=True
    ):
        super().__init__(dist, up, zoom)
        # Reminder: these width and height represent the view-plane in world coordinates
        # and has little to do with the canvas/viewport size.
        self.width = width
        self.height = height
        self.maintain_aspect = maintain_aspect

        self.set_view_size(1, 1)
        self.update_projection_matrix()

    def __repr__(self) -> str:
        return (
            f"OrthographicCamera({self.width}, {self.height}, {self.dist}, {self.up})"
        )

    @property
    def width(self):
        """The (minimum) width of the view-cube."""
        return self._width

    @width.setter
    def width(self, value):
        self._width = float(value)

    @property
    def height(self):
        """The (minimum) height of the view-cube."""
        return self._height

    @height.setter
    def height(self, value):
        self._height = float(value)

    @property
    def maintain_aspect(self):
        """Whether the aspect ration is maintained as the window size changes."""
        return self._maintain_aspect

    @maintain_aspect.setter
    def maintain_aspect(self, value):
        self._maintain_aspect = bool(value)

    def _get_near_and_far_plane(self):
        d = self._dist
        return -500 * d, 500 * d

    def set_view_size(self, width, height):
        self._view_aspect = width / height

    def update_projection_matrix(self):
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

        bottom = -0.5 * height
        top = +0.5 * height
        left = -0.5 * width
        right = +0.5 * width
        near, far = self._get_near_and_far_plane()
        # Set matrices
        # The linalg ortho projection puts xyz in the range -1..1, but
        # in the coordinate system of wgpu (and this lib) the depth is
        # expressed in 0..1, so we also correct for that.
        self.projection_matrix.make_orthographic(left, right, top, bottom, near, far)
        self.projection_matrix.premultiply(
            Matrix4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 1)
        )
        self.projection_matrix_inverse.get_inverse(self.projection_matrix)
