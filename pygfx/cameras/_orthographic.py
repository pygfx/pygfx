from ._perspective import PerspectiveCamera


class OrthographicCamera(PerspectiveCamera):
    """An orthographic camera, useful for orthograpic views and 2D content.

    This is really just a PerspectiveCamera with the fov locked to zero.

    Parameters
    ----------
    width : float
        The (minimum) width of the view-cube. The actual view
        may be wider if the viewport is relatively wide.
    height : float
        The (minimum) height of the view-cube. The actual view
        may be height if the viewport is relatively heigh.
    zoom: float
        An additional zoom factor, equivalent to attaching a zoom lens.
    maintain_aspect : bool
        Whether the aspect ration is maintained as the window size changes.
        Default True. If false, the dimensions are stretched to fit the window.

    """

    _fov_range = 0, 0

    def __init__(self, width=1, height=1, *, zoom=1, maintain_aspect=True):
        super().__init__(0, zoom=zoom, maintain_aspect=maintain_aspect)
        self.width = width
        self.height = height
