from ._perspective import PerspectiveCamera


class OrthographicCamera(PerspectiveCamera):
    """An orthographic camera, useful for orthograpic views and 2D content.

    Technically, this is a PerspectiveCamera with the fov locked to zero.

    Parameters
    ----------
    width: float
        The (minimum) width of the view-cube. The actual view
        may be wider if the viewport is relatively wide.
    height: float
        The (minimum) height of the view-cube. The actual view
        may be taller if the viewport is relatively high.
    zoom: float
        An additional zoom factor, equivalent to attaching a zoom lens.
    maintain_aspect: bool
        Whether the aspect ration is maintained as the window size changes.
        Default True. If false, the dimensions are stretched to fit the window.
    depth_range: 2-tuple
        The values for the near and far clipping planes. If not given
        or None, the clip planes will be calculated automatically based
        on the fov, width, and height.
    """

    _fov_range = 0, 0

    def __init__(
        self, width=1, height=1, *, zoom=1, maintain_aspect=True, depth_range=None
    ):
        super().__init__(
            0,
            None,
            width=width,
            height=height,
            zoom=zoom,
            maintain_aspect=maintain_aspect,
            depth_range=depth_range,
        )
