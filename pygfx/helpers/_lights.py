import math

import numpy as np

from ..objects import Light
from ..utils.transform import AffineBase, callback

from .. import (
    sphere_geometry,
    Mesh,
    MeshBasicMaterial,
    Geometry,
    LineArrowMaterial,
    LineSegmentMaterial,
    Line,
)


class PointLightHelper(Mesh):
    """Light source indicator for point lights.

    Helper class to visualize a point light using some geometry (sphere by
    default). The helper object should be a parent of the light object.

    Parameters
    ----------
    size : float
        If geometry is None, the size of the indicator. Ignored otherwise.
    geometry : Geometry
        The geometry used to visualize the location of the light source. If None,
        a sphere will be used.
    color : Color
        The color of the sphere. If None, match the color of the light.

    """

    def __init__(self, size=1, geometry=None, color=None):
        assert isinstance(size, (int, float))
        if geometry is None:
            geometry = sphere_geometry(size)
        self._color = color
        material = MeshBasicMaterial(color="#fff")
        super().__init__(geometry, material)

        self.world.on_update(self._update)

    @callback
    def _update(self, transform: AffineBase) -> None:
        if self._color is None and isinstance(self.parent, Light):
            color = self.parent.color
            if color != self.material.color:
                self.material.color = color


class DirectionalLightHelper(Line):
    """Light source indicator for directional lights.

    Helper class to visualize a directional light. It shows arrows eminating
    from the light's position. If show_shadow_extent is True, it also shows the
    vector to the target and the extent of the shadowmap. The helper object
    should be a child of the light object.

    Parameters
    ----------
    ray_length : float
        The length of the indicator arrows.
    color : Color
        The color of the arrows. If None, match the color of the light.
    show_shadow_extent : bool
        If True, indicate the extent of the shadow map.

    """

    def __init__(self, ray_length=1, color=None, show_shadow_extent=False):
        self._color = color

        super().__init__(
            Geometry(positions=np.zeros((8, 3), np.float32)),
            LineArrowMaterial(color="#fff", thickness=5),
        )

        self._shadow_helper = Line(
            Geometry(positions=np.zeros((14, 3), np.float32)), LineSegmentMaterial()
        )
        self.add(self._shadow_helper)

        self.ray_length = ray_length
        self.show_shadow_extent = show_shadow_extent

        self.world.on_update(self._update)

    @property
    def ray_length(self):
        """The length of the arrows indicating light rays."""
        return self._ray_length

    @ray_length.setter
    def ray_length(self, value):
        self._ray_length = float(value)

        # Update geometry
        length = self._ray_length
        len5 = length / 5
        positions = np.array(
            [
                [len5, 0, 0],
                [len5, 0, -length],
                [-len5, 0, 0],
                [-len5, 0, -length],
                [0, len5, 0],
                [0, len5, -length],
                [0, -len5, 0],
                [0, -len5, -length],
            ],
            np.float32,
        )
        self.geometry.positions.data[:] = positions
        self.geometry.positions.update_range(0, 8)

    @property
    def show_shadow_extent(self):
        """Whether to also show the extent of the shadowmap."""
        return self._show_shadow_extent

    @show_shadow_extent.setter
    def show_shadow_extent(self, value):
        self._show_shadow_extent = bool(value)
        self._shadow_helper.visible = self._show_shadow_extent

    @callback
    def _update(self, transform: AffineBase):
        if not isinstance(self.parent, Light):
            return

        if self._color is None:
            color = self.parent.color
            if color != self.material.color:
                self.material.color = color
                self._shadow_helper.material.color = color

        half_w = self.parent.shadow.camera.width / 2
        half_h = self.parent.shadow.camera.height / 2
        cur_size = np.abs(self._shadow_helper.geometry.positions.data[0])
        ref_size = (half_w, half_h, 0)

        if not np.isclose(cur_size, ref_size).all():
            positions = np.array(
                [
                    # Square
                    [-half_w, half_h, 0],
                    [half_w, half_h, 0],
                    [half_w, half_h, 0],
                    [half_w, -half_h, 0],
                    [half_w, -half_h, 0],
                    [-half_w, -half_h, 0],
                    [-half_w, -half_h, 0],
                    [-half_w, half_h, 0],
                    # Diagonals
                    [-half_w, -half_h, 0],
                    [half_w, half_h, 0],
                    [half_w, -half_h, 0],
                    [-half_w, half_h, 0],
                ],
                np.float32,
            )
            self._shadow_helper.geometry.positions.data[:12] = positions
            self._shadow_helper.geometry.positions.update_range(0, 12)

        lastval = -self.parent._gfx_distance_to_target
        if not np.isclose(lastval, self._shadow_helper.geometry.positions.data[13, 2]):
            self._shadow_helper.geometry.positions.data[13] = (0, 0, lastval)
            self._shadow_helper.geometry.positions.update_range(13, 14)


class SpotLightHelper(Line):
    """Light source indicator for spot lights.

    Helper class to visualize a spot light. The helper object should be a child
    of the light object.

    Parameters
    ----------
    color : Color
        The color of the arrows. If None, match the color of the light.

    """

    def __init__(self, color=None):
        self._color = color

        positions = [
            [0, 0, 0],
            [0, 0, -1],
            [0, 0, 0],
            [1, 0, -1],
            [0, 0, 0],
            [-1, 0, -1],
            [0, 0, 0],
            [0, 1, -1],
            [0, 0, 0],
            [0, -1, -1],
        ]

        for i in range(32):
            p1 = i / 32 * math.pi * 2
            p2 = (i + 1) / 32 * math.pi * 2

            positions.append([math.cos(p1), math.sin(p1), -1])
            positions.append([math.cos(p2), math.sin(p2), -1])

        super().__init__(
            Geometry(positions=positions),
            LineSegmentMaterial(thickness=1.0),
        )
        self.update_id = self.world.on_update(self._update)

    @callback
    def _update(self, transform: AffineBase):
        if not isinstance(self.parent, Light):
            return
        light = self.parent

        if self._color is None:
            color = light.color
            if color != self.material.color:
                self.material.color = color

        cone_length = light.distance or 1000
        cone_width = cone_length * math.tan(light.angle)

        # Temporarily remove callback to avoid infinite recursion when setting `self.local.scale`.
        self.world.remove_callback(self.update_id)
        self.local.scale = (cone_width, cone_width, cone_length)
        self.update_id = self.world.on_update(self._update)
