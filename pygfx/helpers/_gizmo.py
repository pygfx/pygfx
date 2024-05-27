"""
A transform gizmo to manipulate world objects.
"""

import numpy as np
import pylinalg as la

from ..objects import Line, Mesh
from ..geometries import Geometry, sphere_geometry, cone_geometry, box_geometry
from ..materials import MeshBasicMaterial, LineMaterial
from ..objects import WorldObject
from ..utils.viewport import Viewport
from ..utils.transform import AffineTransform, mat_inv


# Colors in hsluv space - https://www.hsluv.org/
# With H: 0/120/240, S: 100, L: 50
WHITE = "#ffffff"
RED = "#ea0064"
GREEN = "#3f8700"
BLUE = "#007eb7"

THICKNESS = 3


class TransformGizmo(WorldObject):
    """Gizmo to manipulate a WorldObject.

    This Gizmo allows to interactively control WorldObjects with the mouse
    inside a canvas. It can translate and rotate objects relative to the world
    frame, local frame, or camera frame (screen space).

    To control the Gizmo:

    * Click the center sphere to toggle between object-space, world-space and
      screen-space.
    * Grab the center sphere for uniform scaling.
    * Grab the cubes for one-directional scaling (only in object-space).
    * Grab the arrows to translate in one direction.
    * Grab the planes to translate in two directions.
    * Grab the spheres to rotate.

    Parameters
    ----------
    object : WorldObject
        The controlled world object.
    screen_size : float
        The approximate size of the widget in logical pixels. Default 100.

    """

    def __init__(self, object=None, screen_size=100):
        super().__init__()
        self._object_to_control = None

        # We store these as soon as we get a call in ``add_default_event_handlers``
        self._viewport = None
        self._camera = None
        self._ndc_to_screen = None

        # A dict that stores the state at the start of a drag. Or None.
        self._ref = None
        self.gizmo_scale = np.ones(3, dtype=float)

        # The radius (in pixels) the gizmo's screen-space bounding box should occupy
        # (aka. the desired on-screen size of the gizmo)
        self._screen_size = float(screen_size)

        # the extent of the gizmo measured along each cardinal direction
        # expressed in the local frame
        # order: right, up, forward, left, down, backward (first 3-tuple is
        # along positive (x, y, z), second 3-tuple is along negative (x, y, z))
        self._local_extents = None

        # Init
        self._create_elements()
        self.toggle_mode("object")  # object, world, screen
        self._highlight()
        self.set_object(object)

    def set_object(self, object: WorldObject):
        """Update the controlled object.

        Parameters
        ----------
        object : WorldObject
            The new controlled object.

        """

        if not isinstance(object, (WorldObject, None)):
            raise ValueError("The object must be None or a WorldObject instance.")

        self._object_to_control = object

    def toggle_mode(self, mode=None):
        """Switch the reference frame.

        Parameters
        ----------
        mode : str
            The reference frame to switch to. Must be one of  ``"object"``
            (local frame), ``"world"`` (world frame), or ``"screen"`` (camera
            frame). If None the next mode (following this order) is selected.

        """

        modes = "object", "world", "screen"
        if not mode:
            mode = {"object": "world", "world": "screen"}.get(self._mode, "object")
        if mode not in modes:
            raise ValueError(f"Invalid mode '{mode}', must be one of {modes}.")
        self._mode = mode
        self._on_mode_switch()  # The gizmo looks a bit different in each mode

    def _create_elements(self):
        """Create lines, handles, etc."""

        # The location of the scale handles, and where the arcs meet the lines.
        self._halfway = halfway = 0.65

        # --- the parts that are at the center

        # Create screen translate handle
        sphere_geo = sphere_geometry(0.07)
        scale_uniform = Mesh(
            sphere_geo,
            MeshBasicMaterial(color=WHITE, pick_write=True),
        )
        scale_uniform.local.scale = (1.5, 1.5, 1.5)

        # --- the parts that are fully in one dimension

        # Create lines
        line_geo = Geometry(positions=[(0, 0, 0), (halfway, 0, 0)])
        line_x = Line(
            line_geo,
            LineMaterial(thickness=THICKNESS, color=RED),
        )
        line_y = Line(
            line_geo,
            LineMaterial(thickness=THICKNESS, color=GREEN),
        )
        line_z = Line(
            line_geo,
            LineMaterial(thickness=THICKNESS, color=BLUE),
        )

        # Create 1D translate handles
        cone_geo = cone_geometry(0.1, 0.17)
        cone_geo.positions.data[:] = cone_geo.positions.data[:, ::-1]  # xyz->zyx
        translate_x = Mesh(
            cone_geo,
            MeshBasicMaterial(color=RED, pick_write=True),
        )
        translate_y = Mesh(
            cone_geo,
            MeshBasicMaterial(color=GREEN, pick_write=True),
        )
        translate_z = Mesh(
            cone_geo,
            MeshBasicMaterial(color=BLUE, pick_write=True),
        )
        translate_x.local.position = (1, 0, 0)
        translate_y.local.position = (0, 1, 0)
        translate_z.local.position = (0, 0, 1)

        # Create scale handles
        cube_geo = box_geometry(0.1, 0.1, 0.1)
        scale_x = Mesh(
            cube_geo,
            MeshBasicMaterial(color=RED, pick_write=True),
        )
        scale_y = Mesh(
            cube_geo,
            MeshBasicMaterial(color=GREEN, pick_write=True),
        )
        scale_z = Mesh(
            cube_geo,
            MeshBasicMaterial(color=BLUE, pick_write=True),
        )
        scale_x.local.position = (halfway, 0, 0)
        scale_y.local.position = (0, halfway, 0)
        scale_z.local.position = (0, 0, halfway)

        # --- the parts that are in a plane

        # Create arcs
        t = np.linspace(0, np.pi / 2, 64)
        arc_positions = np.stack([0 * t, np.sin(t), np.cos(t)], 1).astype(np.float32)
        arc_geo = Geometry(positions=arc_positions * halfway)
        arc_yz = Line(
            arc_geo,
            LineMaterial(thickness=THICKNESS, color=RED),
        )
        arc_zx = Line(
            arc_geo,
            LineMaterial(thickness=THICKNESS, color=GREEN),
        )
        arc_xy = Line(
            arc_geo,
            LineMaterial(thickness=THICKNESS, color=BLUE),
        )

        # Create in-plane translate handles
        plane_geo = box_geometry(0.01, 0.15, 0.15)
        translate_yz = Mesh(
            plane_geo,
            MeshBasicMaterial(color=RED, pick_write=True),
        )
        translate_zx = Mesh(
            plane_geo,
            MeshBasicMaterial(color=GREEN, pick_write=True),
        )
        translate_xy = Mesh(
            plane_geo,
            MeshBasicMaterial(color=BLUE, pick_write=True),
        )
        inside_arc = 0.4 * halfway
        translate_yz.local.position = (0, inside_arc, inside_arc)
        translate_zx.local.position = (inside_arc, 0, inside_arc)
        translate_xy.local.position = (inside_arc, inside_arc, 0)

        # Create rotation handles
        # These are positioned on each mode switch
        rotate_yz = Mesh(
            sphere_geo,
            MeshBasicMaterial(color=RED, pick_write=True),
        )
        rotate_zx = Mesh(
            sphere_geo,
            MeshBasicMaterial(color=GREEN, pick_write=True),
        )
        rotate_xy = Mesh(
            sphere_geo,
            MeshBasicMaterial(color=BLUE, pick_write=True),
        )

        # --- post-process

        # Rotate objects to their correct orientation
        ob: WorldObject
        for ob in [line_y, translate_y, scale_y, translate_zx, rotate_zx]:
            ob.local.rotation = la.quat_from_axis_angle((0, 0, 1), np.pi / 2)
        for ob in [line_z, translate_z, scale_z, translate_xy, rotate_xy]:
            ob.local.rotation = la.quat_from_axis_angle((0, -1, 0), np.pi / 2)

        arc_xy.local.rotation = la.quat_from_axis_angle((0, 1, 0), np.pi / 2)
        arc_zx.local.rotation = la.quat_from_axis_angle((0, 0, 1), -np.pi / 2)

        # Store the objects
        self._center_sphere = scale_uniform
        self._line_children = line_x, line_y, line_z
        self._arc_children = arc_yz, arc_zx, arc_xy
        self._translate1_children = translate_x, translate_y, translate_z
        self._translate2_children = translate_yz, translate_zx, translate_xy
        self._translate_children = self._translate1_children + self._translate2_children
        self._scale_children = scale_x, scale_y, scale_z
        self._rotate_children = rotate_yz, rotate_zx, rotate_xy

        # Lines and arcs are never opaque. That way they're also not
        # pickable, so they won't be "in the way".
        for ob in self._line_children + self._arc_children:
            ob.material.opacity = 0.6

        # Assign dimensions
        scale_uniform.dim = None
        for triplet in [
            self._line_children,
            self._translate1_children,
            self._scale_children,
            self._rotate_children,
        ]:
            for i, ob in enumerate(triplet):
                ob.dim = i
        for i, ob in enumerate(self._translate2_children):
            dim = [0, 1, 2]
            dim.pop(i)
            ob.dim = tuple(dim)

        # Attach to the gizmo object
        self.add(*self._line_children)
        self.add(*self._arc_children)
        self.add(self._center_sphere)
        self.add(*self._translate_children)
        self.add(*self._scale_children)
        self.add(*self._rotate_children)

        # work out local extent
        self._local_extents = np.empty((6, 3), dtype=float)
        scales = self.get_bounding_box().ravel()  # we know our bb is not None
        self._local_extents[:3] = np.diag(scales[3:])
        self._local_extents[3:] = np.diag(scales[:3])

    def _on_mode_switch(self):
        """When the mode is changed, some adjustments are made."""

        # Note: the elements are affected by the mode in various ways.
        # Much of that (e.g. visibility) is already handled in the
        # _update_xx functions, so we don't have to do that here.

        # Update position of rotation handles
        rotate_yz, rotate_zx, rotate_xy = self._rotate_children
        halfway = self._halfway
        on_arc = halfway * 2**0.5 / 2
        rotate_yz.local.position = (0, on_arc, on_arc)
        rotate_zx.local.position = (on_arc, 0, on_arc)
        rotate_xy.local.position = (on_arc, on_arc, 0)

    def _highlight(self, object=None):
        """Change the appearance during interaction for visual feedback
        on what is being manipulated.
        """
        # Reset thickness of all lines
        for ob in self._line_children:
            if ob.material.thickness != THICKNESS:
                ob.material.thickness = THICKNESS

        for ob in self._arc_children:
            if ob.material.thickness != THICKNESS:
                ob.material.thickness = THICKNESS

        # Set thickness of lines corresponding to the object
        if object:
            lines = []
            dim = object.dim
            if dim is None:  # center handle
                pass
            elif isinstance(dim, int):
                if object in self._rotate_children:
                    lines.append(self._arc_children[dim])
                else:  # translate1 or scale
                    lines.append(self._line_children[dim])
            else:  # translate2
                lines.append(self._line_children[dim[0]])
                lines.append(self._line_children[dim[1]])
            for ob in lines:
                ob.material.thickness *= 1.75

    def update_gizmo(self, event):
        if event.type != "before_render":
            return

        # Note that we almost always update the transform (scale,
        # rotation, position) which means the matrix changed, and so
        # does the world_matrix of all children. In effect the uniforms
        # of all elements need updating anyway, so any other changes
        # to wobject properties (e.g. visibility) are "free" - no need
        # to only update if it actually changes.
        if not self._object_to_control:
            self.visible = False
        elif self._viewport and self._camera:
            self.visible = True
            self._update_ndc_screen_transform()
            self._update_directions()
            self._update_gizmo_transform()
            self._update_visibility()

    def _update_ndc_screen_transform(self):
        # Note: screen origin is at top left corner of NDC with Y-axis pointing down
        x_dim, y_dim = self._viewport.logical_size
        screen_space = AffineTransform()
        screen_space.position = (-1, 1, 0)
        screen_space.scale = (2 / x_dim, -2 / y_dim, 1)
        self._ndc_to_screen = screen_space.inverse_matrix
        self._screen_to_ndc = screen_space.matrix

    def _update_directions(self):
        """
        Calculate how much 1 unit of translation in the draggable space (aka
        mode) translates the object in world and screen space.

        """

        # work out the transforms between the spaces
        camera = self._camera
        if self._mode == "object":
            # local space is draggable
            local_to_world = self._object_to_control.world.matrix
            local_to_ndc = camera.camera_matrix @ local_to_world
            local_to_screen = self._ndc_to_screen @ local_to_ndc
        elif self._mode == "world":
            # world space is draggable
            local_to_world = np.eye(4)
            local_to_ndc = camera.camera_matrix @ local_to_world
            local_to_screen = self._ndc_to_screen @ local_to_ndc
        elif self._mode == "screen":
            # camera space is draggable
            local_to_world = camera.world.matrix
            local_to_ndc = camera.projection_matrix
            local_to_screen = self._ndc_to_screen @ local_to_ndc
        else:  # This cannot happen, in theory
            raise RuntimeError(f"Unexpected mode: `{self._mode}`")

        # points referring to local coordinate axes and origin
        local_points = np.zeros((4, 3))
        local_points[1:, :] = np.eye(3)
        if self._mode == "screen":
            # reference frame has a z-offset from screen origin
            object_to_ndc = camera.camera_matrix @ self._object_to_control.world.matrix
            depth = la.vec_transform((0, 0, 0), object_to_ndc)[2]

            local_points[3, 2] = -1  # camera has inverted Z axis
            local_points[:, 2] -= depth

        # express unit vectors and origin in the various frames
        world_points = la.vec_transform(local_points, local_to_world)
        ndc_points = la.vec_transform(local_points, local_to_ndc)
        screen_points = la.vec_transform(local_points, local_to_screen)

        # store the directions for future use
        self._world_directions = world_points[1:] - world_points[0]
        self._ndc_directions = ndc_points[1:] - ndc_points[0]
        self._screen_directions = screen_points[1:] - screen_points[0]

    def _update_gizmo_transform(
        self,
    ):
        """

        Set the gizmo's transform to keep it in sync with the object it is
        tracking while accounting for the gizmo's mode.

        Position: Set to match the tracked object.
        Rotation: Set to indicate translation directions of the current mode.
        Scale: Set to have the target on-screen size.

        Note: This function also flips the directions of the gizmo's local axes
        so that handles always point towards the camera.
        """

        self.world.position = self._object_to_control.world.position

        # negative/flipped scale on 2 axes registers as 180° rotation. This will
        # be undone by the rotation update below and will sometimes flip the
        # gizmo during rotation updates. Instead, reset scale and restore the
        # desired scale after the rotation update.
        self.world.scale = 1

        if self._mode == "object":
            self.world.rotation = self._object_to_control.world.rotation
        elif self._mode == "world":
            self.world.rotation = np.array((0, 0, 0, 1))
        else:  # self._mode == "screen"
            self.world.rotation = self._camera.world.rotation

        # During interaction we don't update gzimo scale and axis flip, this way:
        # * During a rotation the gizmo does not flip,
        #   making it easier to see how much was rotated.
        # * During a translation the gizmo keeps its "world size",
        #   so that the perspective helps you see how the gizmo has moved.
        if self._ref:
            self.world.scale = self.gizmo_scale
            return

        # size on screen for scale=1
        local_to_screen = (
            self._ndc_to_screen @ self._camera.camera_matrix @ self.world.matrix
        )
        screen_extents = la.vec_transform(self._local_extents, local_to_screen)
        origin_screen = la.vec_transform((0, 0, 0), local_to_screen)
        screen_directions = screen_extents - origin_screen

        # radius of bounding circle (in screen space) and scaling to set to
        # desired radius (aka _screen_size)
        size_1_radius = np.max(np.linalg.norm(screen_directions[:, :2], axis=-1))
        self.gizmo_scale[:] = self._screen_size / size_1_radius

        # if required, flip axes so that handles always point towards the camera
        eps = 1e-10
        should_flip = screen_directions[:3, 2] > eps
        self.gizmo_scale *= 1 - 2 * should_flip

        self.world.scale = self.gizmo_scale

    def _update_visibility(self):
        """Depending on the mode and the orientation of the camera,
        some elements are made invisible.
        """

        # compute the viewing angle onto the gizmo's coordinate planes
        screen_normal = la.vec_transform((0, 0, -1), self._camera.world.matrix)
        screen_normal = la.vec_normalize(screen_normal)
        plane_normal = (
            la.vec_transform(np.eye(3), self.world.matrix) - self.world.position
        )
        plane_normal = la.vec_normalize(plane_normal)
        cos_angle = np.sum(plane_normal * screen_normal, axis=-1)
        viewing_angle = np.pi / 2 - np.arccos(np.clip(np.abs(cos_angle), 0, 1))

        # compute the size of the gizmo's axes on screen
        gizmo_to_screen = (
            self._ndc_to_screen @ self._camera.camera_matrix @ self.world.matrix
        )
        origin_screen = la.vec_transform((0, 0, 0), gizmo_to_screen)
        axes_screen = la.vec_transform(np.eye(3), gizmo_to_screen) - origin_screen
        ax_size = np.linalg.norm(axes_screen[:, :2], axis=-1)

        # check which handles should be shown
        show_1d_translation = ax_size > 30
        show_1d_scaling = ax_size > 30
        show_2d_translation = viewing_angle > deg_to_rad(15)

        if self._mode == "screen":
            show_1d_translation[2] = False
            show_2d_translation[:] = (False, False, True)

        # Per-dimension scaling is only possible in object-mode
        if self._mode != "object":
            show_1d_scaling[:] = False

        # set the visibility
        # Note: uniform scale and rotations are always visible
        for dim in (0, 1, 2):
            self._translate1_children[dim].visible = show_1d_translation[dim]
            self._scale_children[dim].visible = show_1d_scaling[dim]
            self._translate2_children[dim].visible = show_2d_translation[dim]

    # %% Event handling

    def add_default_event_handlers(self, viewport, camera):
        """Register Gizmo callbacks."""

        # Store objects that we need outside the event handling. In
        # contrast to e.g. a controller, the Gizmo also needs to do
        # some calculation at draw time (or right before a draw, to be
        # precise), and for these calculations it needs the viewport
        # and camera.
        viewport = Viewport.from_viewport_or_renderer(viewport)
        self._viewport = viewport
        self._camera = camera

        self.add_event_handler(
            self.process_event,
            "pointer_down",
            "pointer_move",
            "pointer_up",
            "wheel",
        )

        viewport.renderer.add_event_handler(self.update_gizmo, "before_render")

    def process_event(self, event):
        """Callback to handle gizmo-related events."""

        # No interaction if there is no object to control
        if not self._object_to_control:
            return

        # Triage over event type
        type = event.type

        if type == "pointer_down":
            if event.button != 1 or event.modifiers:
                return
            self._ref = None
            # NOTE: I imagine that if multiple tools are active, they
            # each ask for picking info, causing multiple buffer reads
            # for the same location. However, with the new event system
            # this is probably not a problem, when wobjects receive events.
            ob = event.target
            if ob not in self.children:
                return
            # Depending on the object under the pointer, we scale/translate/rotate
            if ob == self._center_sphere:
                self._handle_start("scale", event, ob)
            elif ob in self._translate_children:
                self._handle_start("translate", event, ob)
            elif ob in self._scale_children:
                self._handle_start("scale", event, ob)
            elif ob in self._rotate_children:
                self._handle_start("rotate", event, ob)
            # Highlight the object
            self._highlight(ob)
            self._viewport.renderer.request_draw()
            self.set_pointer_capture(event.pointer_id, event.root)

        elif type == "pointer_up":
            if not self._ref:
                return
            if self._ref["dim"] is None and self._ref["maxdist"] < 3:
                self.toggle_mode()  # clicked on the center sphere
            self._ref = None
            # De-highlight the object
            self._highlight()
            self._viewport.renderer.request_draw()

        elif type == "pointer_move":
            if not self._ref:
                return
            # Get how far we've moved from starting point - we have a dead zone
            dist = (
                (event.x - self._ref["event_pos"][0]) ** 2
                + (event.y - self._ref["event_pos"][1]) ** 2
            ) ** 0.5
            self._ref["maxdist"] = max(self._ref["maxdist"], dist)
            # Delegate to the correct handler
            if self._ref["maxdist"] < 3:
                pass
            elif self._ref["kind"] == "translate":
                self._handle_translate_move(event)
            elif self._ref["kind"] == "scale":
                self._handle_scale_move(event)
            elif self._ref["kind"] == "rotate":
                self._handle_rotate_move(event)
            # Keep viz up to date
            self._viewport.renderer.request_draw()

    def _handle_start(self, kind, event, ob: WorldObject):
        """Initiate a drag. We create a snapshot of the relevant state at this point."""
        this_pos = self._object_to_control.world.position
        ob_pos = ob.world.position
        self._ref = {
            "kind": kind,
            "event_pos": np.array((event.x, event.y)),
            "dim": ob.dim,
            "maxdist": 0,
            # Transform at time of start
            "pos": self._object_to_control.world.position,
            "scale": self._object_to_control.world.scale,
            "rot": self._object_to_control.world.rotation,
            "world_pos": ob_pos,
            "world_offset": ob_pos - this_pos,
            "ndc_pos": la.vec_transform(ob_pos, self._camera.camera_matrix),
            # Gizmo direction state at start-time of drag
            "flips": np.sign(self.world.scale),
            "world_directions": self._world_directions.copy(),
            "ndc_directions": self._ndc_directions.copy(),
            "screen_directions": self._screen_directions.copy(),
        }

    def _handle_translate_move(self, event):
        """Translate action, either using a translate1 or translate2 handle."""

        world_to_screen = self._ndc_to_screen @ self._camera.camera_matrix
        screen_to_world = mat_inv(world_to_screen)

        if isinstance(self._ref["dim"], int):
            travel_directions = (self._ref["dim"],)
        else:
            travel_directions = self._ref["dim"]

        screen_travel = np.array(
            (
                event.x - self._ref["event_pos"][0],
                event.y - self._ref["event_pos"][1],
            )
        )

        # units dragged along gizmo axes
        screen_directions = self._ref["screen_directions"][travel_directions, :]
        if len(screen_directions) == 1:
            # translate 1D: only count movement along translation axis
            units_traveled = get_scale_factor(screen_directions[..., :2], screen_travel)
        else:
            # translate 2D: change basis from screen to gizmo axes
            screen_to_axes = mat_inv(screen_directions[..., :2].T)
            units_traveled = screen_to_axes @ screen_travel

        # pixel units to world units
        # Note: location of translation matters because perspective cameras have
        # shear, i.e., we need to account for start
        start = la.vec_transform(self._ref["world_pos"], world_to_screen)
        end = start + screen_directions.T @ units_traveled
        end_world = la.vec_transform(end, screen_to_world)
        world_units_traveled = end_world - self._ref["world_pos"]

        self._object_to_control.world.position = self._ref["pos"] + world_units_traveled

    def _handle_scale_move(self, event):
        """Scale action."""
        # Get dimension
        dim = self._ref["dim"]

        # Get how the mouse has moved
        screen_moved = np.array(
            (
                event.x - self._ref["event_pos"][0],
                event.y - self._ref["event_pos"][1],
                0,
            )
        )

        if dim is None:
            # Uniform scale
            ref_vec = np.array((1, -1, 0))
            factor = get_scale_factor(ref_vec, screen_moved)
            scale = 2 ** (factor / 100)
            scale = np.array((scale, scale, scale))
        else:
            # Get how much the mouse has moved in the ref direction
            screen_dir = self._ref["screen_directions"][dim]
            factor = get_scale_factor(screen_dir, screen_moved)

            # we flip gizmo axis to point towards the user. As a result,
            # users expect the direction of positive scaling to flip, too
            is_flipped = self.local.scale[dim] < 0
            factor *= 1 - 2 * (is_flipped)

            npixels = factor * np.linalg.norm(screen_dir)
            # Calculate the relative scale
            scale = [1, 1, 1]
            scale[dim] = 2 ** (npixels / 100)
            scale = np.array(scale)

        # Apply
        self._object_to_control.local.scale = scale * self._ref["scale"]

    def _handle_rotate_move(self, event):
        """Rotate action."""

        local_to_screen = (
            self._ndc_to_screen @ self._camera.camera_matrix @ self.world.matrix
        )
        object_screen = la.vec_transform((0, 0, 0), local_to_screen)[:2]

        # amount of cursor rotation around gizmo origin (CCW is positive)
        start_direction = self._ref["event_pos"] - object_screen
        start_angle = np.arctan2(start_direction[1], start_direction[0])
        current_direction = np.array((event.x, event.y)) - object_screen
        current_angle = np.arctan2(current_direction[1], current_direction[0])
        cursor_rotation = current_angle - start_angle

        # axis around which the object rotates
        dim = self._ref["dim"]
        world_axis = self._ref["world_directions"][dim]

        # the cursor rotation is measured around the camera's forward direction
        # (CCW is positive) we need to mirror it if the rotation axis points
        # points the other way (when CW is positive)
        ndc_axis = self._ref["ndc_directions"][dim]
        is_mirrored = 2 * int(ndc_axis[2] > 0) - 1
        cursor_rotation *= is_mirrored

        initial_rotation = self._ref["rot"]
        rotation = la.quat_from_axis_angle(world_axis, cursor_rotation)
        self._object_to_control.world.rotation = la.quat_mul(rotation, initial_rotation)


def get_scale_factor(vec1, vec2):
    """
    Vector project vec2 onto vec1. Aka, figure out how long vec2
    is when measured along vec1.

    This is used, for example, to work out how many units the cursor has
    traveled along a given direction.
    """

    # Note: implementing it like this saves a couple square-roots from
    # normalizing
    return np.sum(vec2 * vec1, axis=-1) / np.sum(vec1**2, axis=-1)


def deg_to_rad(degrees):
    return degrees / 360 * (2 * np.pi)
