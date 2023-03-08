from ._generic import GenericCamera


class PerspectiveCamera(GenericCamera):
    """A 3D perspective camera.

    Parameters
    ----------
    fov: float
        The field of view as an angle in degrees. Default is 50. Higher
        values give a wide-angle lens effect. This value is limited
        between 1 and 179.
    aspect : float
        The desired aspect ratio, which is used to determine the vision pyramid's
        boundaries depending on the viewport size. Common values are 16/9 or 4/3. Default 1.
    extent : float
        A measure for the size of the scene that the camera is
        observing. This is also set by `show_object()`. It is also used
        to set the near and far clipping planes, and controllers use
        it to determine what is being looked at.
    """

    _fov_range = 1, 179

    def __init__(self, fov=50, aspect=1, extent=1):
        super().__init__(fov, aspect, extent)

    def __repr__(self) -> str:
        return f"PerspectiveCamera({self.fov}, {self.aspect}, {self.extent})"
