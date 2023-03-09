from ._generic import GenericCamera


class OrthographicCamera(GenericCamera):
    """An orthographic camera, useful for non-perspective views and 2D content.

    Parameters
    ----------
    width : float
        The (minimum) width of the view-cube. The actual view
        may be wider if the viewport is relatively wide.
    height : float
        The (minimum) height of the view-cube. The actual view
        may be height if the viewport is relatively heigh.
    maintain_aspect : bool
        Whether the aspect ration is maintained as the window size changes. Default True.
        If false, the dimensions are stretched to fit the window.

    """

    _fov_range = 0, 0

    def __init__(self, width, height, maintain_aspect=True):
        super().__init__(0)

        self.width = width
        self.height = height
        self.maintain_aspect = maintain_aspect

    def __repr__(self) -> str:
        return f"OrthographicCamera({self.width}, {self.height})"
