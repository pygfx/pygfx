from ._perspective import PerspectiveCamera


class OrthographicCamera(PerspectiveCamera):
    """An orthographic camera, useful for orthograpic views and 2D content.

    Technically, this is a PerspectiveCamera with the fov locked to zero.
    You can use this instead of a PerspectiveCamera if you have data that does
    not make sense to view with a nonzero fov.

    When used with the show methods and/or a controller, the ``OrthographicCamera``'s
    position does not move along the view-axis (to zoom in/out), unlike the
    ``PerspectiveCamera``, which does this even when fov is 0.

    Parameters
    ----------
    width: float
        The (minimum) width of the view-cube. The actual view
        may be wider if the viewport is relatively wide. Default 1.
    height: float
        The (minimum) height of the view-cube. The actual view
        may be taller if the viewport is relatively high. Default 1.
    zoom: float
        An additional zoom factor, equivalent to attaching a zoom lens.
    maintain_aspect: bool
        Whether the aspect ration is maintained as the window size changes.
        Default True. If false, the dimensions are stretched to fit the window.
    depth_multiplier: float
        The multiplier used to calculate the near and far clip planes.
        The range of depth is the ``depth_extent`` times the ``depth_multiplier``.
        Ignored if ``depth_range`` is set.
    depth_extent: float
        The reference size of the scene in the depth dimension. By default this value is auto-determined
        using ``width`` and ``height``, and it gets set by ``show_pos()``, ``show_object()`` and ``show_rect()``.
        Ignored if ``depth_range`` is set.
    depth_range: 2-tuple
        The explicit values for the near and far clip planes. If not given
        or None (the default), the clip planes ware calculated using
        ``fov``, ``depth_multiplier``, and ``depth_extent``.

    """

    _fov_range = 0, 0  # Disallow fov other than zero

    def __init__(
        self,
        width=1,
        height=1,
        *,
        zoom=1,
        maintain_aspect=True,
        depth_multiplier=1000.0,
        depth_extent=None,
        depth_range=None,
    ):
        super().__init__(
            0,
            None,
            width=width,
            height=height,
            zoom=zoom,
            maintain_aspect=maintain_aspect,
            depth_multiplier=depth_multiplier,
            depth_extent=depth_extent,
            depth_range=depth_range,
        )
