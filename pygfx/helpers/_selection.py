"""
A selection gizmo to select world objects.
"""

import numpy as np
import pylinalg as la
import pygfx as gfx

from pygfx.objects import WorldObject
from pygfx.utils.viewport import Viewport
from pygfx.utils.transform import AffineTransform


class SelectionGizmo(WorldObject):
    """Gizmo to draw a Selection Box.

    To invoke the Gizmo:

    * Shift-click on the canvas and start dragging to draw a selection rectangle

    Parameters
    ----------
    renderer : Renderer | Viewport
        The renderer or viewport to use for the gizmo.
    camera : Camera
        The camera to use for the gizmo.
    scene : Scene, optional
        The scene to render the gizmo in. This can also be a Group.
    callback_after_drag : callable, optional
        A callback function to call after the selection is completed.
    callback_during_drag : callable, optional
        A callback function to call during the selection, i.e. as we
        drag the selection box.
    modifier : str, optional
        The modifier key to use to activate the gizmo. Default "Shift".
    edge_color : str | tuple, optional
        The color to use for the edge of the selection box.
    fill_color : str | tuple, optional
        The color to use for the fill of the selection box. Set to
        None (default) to disable the fill.
    line_width : float, optional
        The width of the selection box outlines.
    line_style : "solid" | "dashed" | "dotted", optional
        The line style to use for the selection box outlines.
    force_square : bool, optional
        If True, the selection box will be forced to be a square. Default False.
    show_info : bool
        Whether to render text with additional info on the selection box. Default is False.
    leave : bool
        Whether to leave the selection box after the selection is completed. Default False.
    debug : bool
        Whether to print debug information to the console. Default False.

    """

    # Some default parameters
    _info_font_size = 10  # Font size for the info text
    _info_text_fmt = "screen: ({:.0f}, {:.0f})\nworld: ({:.2f}, {:.2f}, {:.2f})"  # Format string for the info text
    _fill_opacity = 0.3  # Opacity of the fill
    _outline_opacity = 0.7  # Opacity of the outline

    def __init__(
        self,
        renderer,
        camera,
        scene=None,
        callback_after_drag=None,
        callback_during_drag=None,
        modifier="Shift",
        edge_color="yellow",
        fill_color=None,
        line_width=1,
        line_style="dashed",
        force_square=False,
        show_info=False,
        leave=False,
        debug=False,
    ):
        assert modifier in ("Shift", "Ctrl", "Alt", None)

        super().__init__()

        # We store these as soon as we get a call in ``add_default_event_handlers``
        self._viewport = Viewport.from_viewport_or_renderer(renderer)
        self._camera = camera
        self._ndc_to_screen = None
        self._scene = scene
        if self._scene:
            self._scene.add(self)

        # Register widget with event handlers
        self.add_default_event_handlers()

        # Init
        self._show_info = show_info
        self._line_style = line_style
        self._line_width = line_width
        self._modifier = modifier
        self._edge_color = edge_color
        self._fill_color = fill_color
        self._force_square = force_square
        self.visible = False
        self._active = False
        self.debug = debug
        self._leave = leave
        self._callback_after = callback_after_drag
        self._callback_during = callback_during_drag
        self._ndc_to_screen = None
        self._screen_to_ndc = None

        # Generate the elements
        self._create_elements()

        # Stores the selection
        self._sel = {}

    @property
    def bounds_world(self):
        """Selection bounds in world coordinates."""
        if not self._sel:
            return None
        sel = np.vstack([self._sel["start_world"], self._sel["end_world"]])
        return np.vstack([sel.min(axis=0), sel.max(axis=0)])

    @property
    def bounds_screen(self):
        """Selection bounds in screen coordinates."""
        if not self._sel:
            return None
        sel = np.vstack([self._sel["start_screen"], self._sel["end_screen"]])
        return np.vstack([sel.min(axis=0), sel.max(axis=0)])

    def _create_elements(self, clear=False):
        """Create selection box elements."""
        # Generate fill
        if self._fill_color:
            self._fill = gfx.Mesh(
                gfx.Geometry(
                    positions=np.array(
                        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)],
                        dtype=np.float32,
                    ),
                    indices=np.array([[0, 1, 2, 3]], dtype=np.int32),
                ),
                gfx.MeshBasicMaterial(color=self._fill_color),
            )
            self._fill.material.opacity = self._fill_opacity
            self.add(self._fill)
        else:
            self._fill = None

        # Generate outline - this is a line where the first and the last point are
        # at the same position.
        if self._edge_color:
            # Translate line style to dash pattern
            dash_pattern = {
                "solid": (),
                "dashed": (5, 2),
                "dotted": (1, 2),
                "dashdot": (5, 2, 1, 2),
            }.get(self._line_style, self._line_style)

            self._outline = gfx.Line(
                gfx.Geometry(
                    positions=np.array(
                        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)],
                        dtype=np.float32,
                    )
                ),
                gfx.LineMaterial(
                    dash_pattern=dash_pattern,
                    thickness=self._line_width,
                    color=self._edge_color,
                ),
            )
            self._outline.material.opacity = self._outline_opacity
            self.add(self._outline)
        else:
            self._outline = None

        # Generate info text
        if self._show_info:
            self._show_info = (
                gfx.Text(
                    gfx.TextGeometry(
                        markdown="",
                        font_size=self._info_font_size,
                        anchor="bottomright",
                        screen_space=True,
                    ),
                    gfx.TextMaterial(
                        color=self._edge_color if self._edge_color else self._fill_color
                    ),
                ),
                gfx.Text(
                    gfx.TextGeometry(
                        markdown="",
                        font_size=self._info_font_size,
                        anchor="topleft",
                        screen_space=True,
                    ),
                    gfx.TextMaterial(
                        color=self._edge_color if self._edge_color else self._fill_color
                    ),
                ),
            )
            self.add(self._show_info[0])
            self.add(self._show_info[1])

    def update_gizmo(self, event):
        """Update the NDC transform."""
        if event.type != "before_render":
            return

        if self._viewport and self._camera and self._active:
            self._update_ndc_screen_transform()

    def _update_ndc_screen_transform(self):
        # Note: screen origin is at top left corner of NDC with Y-axis pointing down
        x_dim, y_dim = self._viewport.logical_size
        screen_space = AffineTransform()
        screen_space.position = (-1, 1, 0)
        screen_space.scale = (2 / x_dim, -2 / y_dim, 1)
        self._ndc_to_screen = screen_space.inverse_matrix
        self._screen_to_ndc = screen_space.matrix

    def add_default_event_handlers(self):
        """Register Gizmo callbacks."""
        self._viewport.renderer.add_event_handler(
            self.process_event,
            "pointer_down",
            "pointer_move",
            "pointer_up",
        )
        # Not sure we actually need to update the gizmo during rendering
        # We could move this to the functions that actually need to map
        # screen to world coordinates and vice versa.
        self._viewport.renderer.add_event_handler(self.update_gizmo, "before_render")

    def process_event(self, event):
        """Callback to handle gizmo-related events."""
        # Triage over event type
        has_mod = self._modifier is None or (self._modifier in event.modifiers)
        if event.type == "pointer_down" and has_mod:
            self._start_drag(event)
            self._viewport.renderer.request_draw()
            # self.set_pointer_capture(event.pointer_id, event.root)

        elif event.type == "pointer_up" and self._active:
            self._stop_drag(event)
            self._viewport.renderer.request_draw()

        elif event.type == "pointer_move" and self._active:
            self._move_selection(event)
            self._viewport.renderer.request_draw()

    def _start_drag(self, event):
        """Initialize the drag."""
        # Set the rectangle to visible
        self.visible = True
        self._active = True
        self._event_modifiers = event.modifiers

        # Set the positions of the selection rectangle
        world_pos = self._screen_to_world((event.x, event.y))

        if self._outline:
            self._outline.geometry.positions.data[:, 0] = world_pos[0]
            self._outline.geometry.positions.data[:, 1] = world_pos[1]
            self._outline.geometry.positions.update_range()
        if self._fill:
            self._fill.geometry.positions.data[:, 0] = world_pos[0]
            self._fill.geometry.positions.data[:, 1] = world_pos[1]
            self._fill.geometry.positions.update_range()

        # In debug mode we will add points
        if self.debug:
            print("Starting at ", world_pos)
            self.remove(*[c for c in self.children if isinstance(c, gfx.Points)])
            point = gfx.Points(
                gfx.Geometry(
                    positions=np.array(
                        [[world_pos[0], world_pos[1], 0]], dtype=np.float32
                    )
                ),
                material=gfx.PointsMaterial(color="r", size=10),
            )
            self.add(point)

        # Store the selection box coordinates
        self._sel = {
            "start_world": world_pos,
            "end_world": world_pos,
            "start_screen": np.array((event.x, event.y)),
            "end_screen": np.array((event.x, event.y)),
        }

        # Update info text (if applicable)
        self._update_info()

    def _stop_drag(self, event):
        """Stop the drag on pointer up."""
        # Set the rectangle to invisible
        self._active = False
        if not self._leave:
            self.visible = False

        if self.debug:
            world_pos = self._screen_to_world((event.x, event.y))
            point = gfx.Points(
                gfx.Geometry(
                    positions=np.array(
                        [[world_pos[0], world_pos[1], 0]], dtype=np.float32
                    )
                ),
                material=gfx.PointsMaterial(color="g", size=10),
            )
            self.add(point)
            print("Stopping with Selection box: ", self._sel)

        if self._callback_after:
            self._callback_after(self)

    def _move_selection(self, event):
        """Translate action, either using a translate1 or translate2 handle."""
        screen_pos = np.array([event.x, event.y])
        if self._force_square:
            dx, dy = screen_pos - self._sel["start_screen"]
            dmin = min(abs(dx), abs(dy))
            screen_pos[0] = self._sel["start_screen"][0] + np.sign(dx) * dmin
            screen_pos[1] = self._sel["start_screen"][1] + np.sign(dy) * dmin

        world_pos = self._screen_to_world(screen_pos)

        if self._outline:
            # The first and the last point on the line remain on the origin
            # The second point goes to (origin, new_y), the third to (new_x, new_y)
            # The fourth to (new_x, origin)
            self._outline.geometry.positions.data[1, 1] = world_pos[1]
            self._outline.geometry.positions.data[2, 0] = world_pos[0]
            self._outline.geometry.positions.data[2, 1] = world_pos[1]
            self._outline.geometry.positions.data[3, 0] = world_pos[0]
            self._outline.geometry.positions.update_range()

        if self._fill:
            self._fill.geometry.positions.data[1, 1] = world_pos[1]
            self._fill.geometry.positions.data[2, 0] = world_pos[0]
            self._fill.geometry.positions.data[2, 1] = world_pos[1]
            self._fill.geometry.positions.data[3, 0] = world_pos[0]
            self._fill.geometry.positions.update_range()

        # Store the selection box coordinates
        self._sel["end_world"] = world_pos
        self._sel["end_screen"] = screen_pos

        # Update info text (if applicable)
        self._update_info()

        if self.debug:
            print("Moving to ", world_pos)
            point = gfx.Points(
                gfx.Geometry(
                    positions=np.array(
                        [[world_pos[0], world_pos[1], 0]], dtype=np.float32
                    )
                ),
                material=gfx.PointsMaterial(color=(1, 1, 1, 0.5), size=10),
            )
            self.add(point)

        if self._callback_during:
            self._callback_during(self)

    def _update_info(self):
        """Update the info text."""
        if not self._show_info:
            return

        # Update the info text
        self._show_info[0].geometry.set_text(
            self._info_text_fmt.format(
                *self._sel["start_screen"], *self._sel["start_world"]
            )
        )
        self._show_info[1].geometry.set_text(
            self._info_text_fmt.format(
                *self._sel["end_screen"], *self._sel["end_world"]
            )
        )

        # Update info text positions
        self._show_info[0].local.position = self._sel["start_world"]
        self._show_info[1].local.position = self._sel["end_world"]

    def _screen_to_world(self, pos):
        """Translate screen positions to world coordinates."""
        if not self._viewport.is_inside(*pos):
            return None

        # Get position relative to viewport
        pos_rel = (
            pos[0] - self._viewport.rect[0],
            pos[1] - self._viewport.rect[1],
        )

        vs = self._viewport.logical_size

        # Convert position to NDC
        x = pos_rel[0] / vs[0] * 2 - 1
        y = -(pos_rel[1] / vs[1] * 2 - 1)
        pos_ndc = (x, y, 0)

        pos_ndc += la.vec_transform(
            self._camera.world.position, self._camera.camera_matrix
        )
        pos_world = la.vec_unproject(pos_ndc[:2], self._camera.camera_matrix)

        return pos_world

    def _world_to_screen(self, pos):
        """Translate world positions to screen coordinates."""
        self._update_ndc_screen_transform()
        world_to_screen = self._ndc_to_screen @ self._camera.camera_matrix

        return la.vec_transform(pos, world_to_screen)

    def is_inside(self, x):
        """Get whether the given positions are inside the selection rect.

        Parameters
        ----------
        x : (N, 3) array |
             Coordinates in world space (i.e. with all transformations applied).

        Returns
        -------
        contained : (N,) bool
            Boolean array indicating whether the coordinates are within the selection.
            Returns None if no selection is present.

        """
        if self.bounds_world is None:
            return None

        assert isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] == 3

        x_screen = self._world_to_screen(x)

        # Check if any of the points are within the selection region
        contained = (
            (x_screen[:, 0] >= self.bounds_screen[0, 0])
            & (x_screen[:, 0] <= self.bounds_screen[1, 0])
            & (x_screen[:, 1] >= self.bounds_screen[0, 1])
            & (x_screen[:, 1] <= self.bounds_screen[1, 1])
        )

        return contained
