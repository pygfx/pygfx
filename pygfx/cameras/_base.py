from time import perf_counter_ns

import numpy as np
import pylinalg as la

from ..objects._base import WorldObject
from ..utils.transform import cached


class Camera(WorldObject):
    """Abstract base camera.

    Camera's are world objects and can be placed in the scene, but this is not required.

    The purpose of a camera is to define the viewpoint for rendering a scene.
    This viewpoint consists of its position and orientation (in the world) and
    its projection.

    In other words, it covers the projection of world coordinates to
    normalized device coordinates (NDC), by the (inverse of) the
    camera's own world matrix and the camera's projection transform.
    The former represent the camera's position, the latter is specific
    to the type of camera.
    """

    _FORWARD_IS_MINUS_Z = True

    def __init__(self):
        super().__init__()
        self._last_modified = perf_counter_ns()

        self._view_size = 1.0, 1.0
        self._view_offset = None

    def flag_update(self):
        self._last_modified = perf_counter_ns()

    @property
    def last_modified(self) -> int:
        return max(self._last_modified, self.world.last_modified)

    def set_view_size(self, width, height):
        """Sets the logical size of the target. Set by the renderer; you should typically not use this."""
        self._view_size = float(width), float(height)
        self.flag_update()

    def set_view_offset(
        self,
        full_width: float,
        full_height: float,
        x: float,
        y: float,
        width: float,
        height: float,
    ):
        """Set the offset in a larger viewing frustrum and override the logical size.

        This is useful for advanced use-cases such as multi-window setups or taking tiled screenshots.
        It is the responsibility of the caller to make sure that the ratio of the ``width`` and ``height``
        match that of the canvas/viewport being rendered to, so that the effective ``pixel_ratio`` is isotropic.

        .. code-block:: python

            # Assuming a canvas with a logical size of 640x480 ...

            # Use a custom logical size
            camera.set_view_offset(320, 240, 0, 0, 320, 240)

            # Render the bottom-left corner, sizes in screen-space become larger (relative to the screen)
            camera.set_view_offset(640, 480, 0, 240, 320, 240)

            # Render the bottom-left corner, sizes in screen-space stay the same (relative to the screen)
            camera.set_view_offset(1280, 960, 0, 480, 640, 480)

        Parameters
        ----------
        full_width (float): The full width of the virtual viewing frustrum.
        full_height (float): The full height of the virtual viewing frustrum.
        x (float): The horizontal offset of the curent sub-view.
        y (float): The vertical offset of the curent sub-view.
        width (float): The width of the current sub-view.
        height (float): The height of the current sub-view.

        """
        # Store values
        self._view_offset = vo = {
            "full_width": float(full_width),
            "full_height": float(full_height),
            "x": float(x),
            "y": float(y),
            "width": float(width),
            "height": float(height),
        }
        # Calculate ndc_offset, a value that can be easily applied in the shader using
        # virtual_ndc = ndc.xy * ndc_offset.xy + ndc_offset.zw
        ax = vo["width"] / vo["full_width"]
        ay = vo["height"] / vo["full_height"]
        self._view_offset["ndc_offset"] = (
            ax,
            ay,
            ax + 2.0 * vo["x"] / vo["full_width"] - 1.0,
            -(ay + 2.0 * vo["y"] / vo["full_height"] - 1.0),
        )
        self.flag_update()

    def clear_view_offset(self):
        """Remove the currently set view offset, returning to a normal view."""
        self._view_offset = None
        self.flag_update()

    def _update_projection_matrix(self) -> np.ndarray:
        raise NotImplementedError()

    def get_state(self):
        """Get the state of the camera as a dict."""
        return {}

    def set_state(self, state):
        """Set the state of the camera from a dict."""
        self.flag_update()

    @property
    def view_matrix(self) -> np.ndarray:
        return self.world.inverse_matrix

    @cached
    def projection_matrix(self) -> np.ndarray:
        base = self._update_projection_matrix()
        if self._view_offset is None:
            return base

        view_offset = self._view_offset
        s_x = view_offset["full_width"] / view_offset["width"]
        s_y = view_offset["full_height"] / view_offset["height"]
        d_x = view_offset["x"] / view_offset["full_width"]
        d_y = view_offset["y"] / view_offset["full_height"]
        t_x = +(s_x - 1.0 - 2.0 * s_x * d_x)
        t_y = -(s_y - 1.0 - 2.0 * s_y * d_y)
        ndc_matrix = np.array(
            [
                [s_x, 0.0, 0.0, t_x],
                [0.0, s_y, 0.0, t_y],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            np.float32,
        )
        proj_matrix = ndc_matrix @ base
        proj_matrix.flags.writeable = False
        return proj_matrix

    @cached
    def projection_matrix_inverse(self) -> np.ndarray:
        proj_inv_matrix = la.mat_inverse(self.projection_matrix)
        proj_inv_matrix.flags.writeable = False
        return proj_inv_matrix

    @cached
    def camera_matrix(self) -> np.ndarray:
        cam_matrix = self.projection_matrix @ self.view_matrix
        cam_matrix.flags.writeable = False
        return cam_matrix


class NDCCamera(Camera):
    """A Camera operating in NDC coordinates.

    Its projection matrix is the identity transform (but its position and rotation can still be set).

    In the NDC coordinate system of wgpu (and Pygfx), x and y are in
    the range -1..1, z is in the range 0..1, and (-1, -1, 0) represents
    the bottom left corner.
    """

    def __init__(self):
        super().__init__()
        self._ndc_proj_matrix = np.eye(4, dtype=float)
        self._ndc_proj_matrix.flags.writeable = False

    def _update_projection_matrix(self):
        return self._ndc_proj_matrix


class ScreenCoordsCamera(Camera):
    """A Camera operating in screen coordinates.

    The depth range is the same as in NDC (0 to 1).
    """

    def __init__(self, invert_y=False):
        super().__init__()
        self._invert_y = bool(invert_y)

    def _update_projection_matrix(self):
        width, height = self._view_size
        sx, sy, sz = 2 / width, 2 / height, 1
        dx, dy, dz = -1, -1, 0
        if self._invert_y:
            dy = -dy
            sy = -sy
        m = sx, 0, 0, dx, 0, sy, 0, dy, 0, 0, sz, dz, 0, 0, 0, 1
        proj_matrix = np.array(m, dtype=float).reshape(4, 4)
        proj_matrix.flags.writeable = False
        return proj_matrix
