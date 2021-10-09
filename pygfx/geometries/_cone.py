import numpy as np

from ._cylinder import CylinderGeometry


class ConeGeometry(CylinderGeometry):
    """A geometry defining a Cone."""

    def __init__(
        self,
        radius=1.0,
        height=1.0,
        radial_segments=8,
        height_segments=1,
        theta_start=0.0,
        theta_length=np.pi * 2,
        open_ended=False,
    ):
        super().__init__(
            radius_bottom=radius,
            radius_top=0.0,
            height=height,
            radial_segments=radial_segments,
            height_segments=height_segments,
            theta_start=theta_start,
            theta_length=theta_length,
            open_ended=open_ended,
        )
