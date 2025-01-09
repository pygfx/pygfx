import numpy as np
import pylinalg as la

from ..objects._base import WorldObject


class Camera(WorldObject):
    """Abstract base camera.

    Camera's are world objects and be placed in the scene, but this is not required.

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

        self._view_size = 1.0, 1.0
        self._view_offset = None

        self.projection_matrix = np.eye(4, dtype=float)
        self.projection_matrix_inverse = np.eye(4, dtype=float)

    def set_view_size(self, width, height):
        """Sets the logical size of the target. Set by the renderer; you should typically not use this."""
        self._view_size = float(width), float(height)

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

    def clear_view_offset(self):
        """Remove the currently set view offset, returning to a normal view."""
        self._view_offset = None

    def update_projection_matrix(self):
        raise NotImplementedError()

    def _finalize_projection_matrix(self, projection_matrix):
        if self._view_offset is None:
            self.projection_matrix = projection_matrix
        else:
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
            self.projection_matrix = ndc_matrix @ projection_matrix

        self.projection_matrix_inverse = la.mat_inverse(self.projection_matrix)

    def get_state(self):
        """Get the state of the camera as a dict."""
        return {}

    def set_state(self, state):
        """Set the state of the camera from a dict."""
        pass

    @property
    def view_matrix(self) -> np.ndarray:
        return self.world.inverse_matrix

    @property
    def camera_matrix(self) -> np.ndarray:
        return self.projection_matrix @ self.view_matrix


class NDCCamera(Camera):
    """A Camera operating in NDC coordinates.

    Its projection matrix is the identity transform (but its position and rotation can still be set).

    In the NDC coordinate system of wgpu (and Pygfx), x and y are in
    the range -1..1, z is in the range 0..1, and (-1, -1, 0) represents
    the bottom left corner.
    """

    def update_projection_matrix(self):
        self._finalize_projection_matrix(np.eye(4))


class ScreenCoordsCamera(Camera):
    """A Camera operating in screen coordinates.

    The depth range is the same as in NDC (0 to 1).
    """

    def update_projection_matrix(self):
        width, height = self._view_size
        sx, sy, sz = 2 / width, 2 / height, 1
        dx, dy, dz = -1, -1, 0
        m = sx, 0, 0, dx, 0, sy, 0, dy, 0, 0, sz, dz, 0, 0, 0, 1
        self._finalize_projection_matrix(np.array(m, np.float32).reshape(4, 4))
