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
from ..utils.transform import callback


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

        # A dict that stores the state at the start of a drag. Or None.
        self._ref = None

        # The (approximate) size of the gizmo on screen
        self._screen_size = float(screen_size)

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

        if self._object_to_control is not None:
            self._object_to_control.world.remove_callback(self.update_gizmo)

        self._object_to_control = object

        if self._object_to_control is not None:
            callback = self.update_gizmo
            self._object_to_control.world.on_update(callback)

            # callback only runs when the object's transform changes, so we need
            # to manually trigger the first time
            callback(self._object_to_control.world)

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
            MeshBasicMaterial(color=WHITE),
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
            MeshBasicMaterial(color=RED),
        )
        translate_y = Mesh(
            cone_geo,
            MeshBasicMaterial(color=GREEN),
        )
        translate_z = Mesh(
            cone_geo,
            MeshBasicMaterial(color=BLUE),
        )
        translate_x.local.position = (1, 0, 0)
        translate_y.local.position = (0, 1, 0)
        translate_z.local.position = (0, 0, 1)

        # Create scale handles
        cube_geo = box_geometry(0.1, 0.1, 0.1)
        scale_x = Mesh(
            cube_geo,
            MeshBasicMaterial(color=RED),
        )
        scale_y = Mesh(
            cube_geo,
            MeshBasicMaterial(color=GREEN),
        )
        scale_z = Mesh(
            cube_geo,
            MeshBasicMaterial(color=BLUE),
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
        arc_zx.local.scale_y = -1
        arc_xy.local.scale_z = -1

        # Create in-plane translate handles
        plane_geo = box_geometry(0.01, 0.15, 0.15)
        translate_yz = Mesh(
            plane_geo,
            MeshBasicMaterial(color=RED),
        )
        translate_zx = Mesh(
            plane_geo,
            MeshBasicMaterial(color=GREEN),
        )
        translate_xy = Mesh(
            plane_geo,
            MeshBasicMaterial(color=BLUE),
        )
        inside_arc = 0.4 * halfway
        translate_yz.local.position = (0, inside_arc, inside_arc)
        translate_zx.local.position = (inside_arc, 0, inside_arc)
        translate_xy.local.position = (inside_arc, inside_arc, 0)

        # Create rotation handles
        # These are positioned on each mode switch
        rotate_yz = Mesh(
            sphere_geo,
            MeshBasicMaterial(color=RED),
        )
        rotate_zx = Mesh(
            sphere_geo,
            MeshBasicMaterial(color=GREEN),
        )
        rotate_xy = Mesh(
            sphere_geo,
            MeshBasicMaterial(color=BLUE),
        )

        # --- post-process

        # Rotate objects to their correct orientation
        ob: WorldObject
        for ob in [line_y, translate_y, scale_y, arc_zx, translate_zx, rotate_zx]:
            ob.local.rotation = la.quaternion_make_from_axis_angle((0, 0, 1), np.pi / 2)
        for ob in [line_z, translate_z, scale_z, arc_xy, translate_xy, rotate_xy]:
            ob.local.rotation = la.quaternion_make_from_axis_angle(
                (0, -1, 0), np.pi / 2
            )

        # Store the objectss
        self._center_sphere = scale_uniform
        self._line_children = line_x, line_y, line_z
        self._arc_children = arc_yz, arc_zx, arc_xy
        self._translate1_children = translate_x, translate_y, translate_z
        self._translate2_children = translate_yz, translate_zx, translate_xy
        self._translate_children = self._translate1_children + self._translate2_children
        self._scale_children = scale_x, scale_y, scale_z
        self._rotate_children = rotate_yz, rotate_zx, rotate_xy

        # Lines and arcs are never apaque. That way they're also not
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

    def _on_mode_switch(self):
        """When the mode is changed, some adjustments are made."""

        # Note: the elements are affected by the mode in various ways.
        # Much of that (e.g. visibility) is already handled in the
        # _update_xx functions, so we don't have to do that here.

        # Update position of rotation handles
        rotate_yz, rotate_zx, rotate_xy = self._rotate_children
        halfway = self._halfway
        on_arc = halfway * 2**0.5 / 2
        if self._mode == "screen":
            rotate_yz.local.position = (0, on_arc, 0)
            rotate_zx.local.position = (on_arc, 0, 0)
            rotate_xy.local.position = (on_arc, on_arc, 0)
        else:
            rotate_yz.local.position = (0, on_arc, on_arc)
            rotate_zx.local.position = (on_arc, 0, on_arc)
            rotate_xy.local.position = (on_arc, on_arc, 0)

    def _highlight(self, object=None):
        """Change the appearance during interaction for visual feedback
        on what is being manipulated.
        """
        # Reset thickness of all lines
        for ob in self._line_children + self._arc_children:
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
                ob.material.thickness = THICKNESS * 1.75

    # %% Updating before each draw

    @callback
    def update_gizmo(self, transform):
        """Update the Gizmo's transform.

        This method is overloaded from the base class to "attach" the gizmo to
        the controlling object.

        """

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
            self._update_directions()
            self._update_scale()
            self._update_visibility()

    def _update_directions(self):
        """Calculate the x/y/z reference directions, which depend on
        mode and camera. Calculate these for world-space, ndc-space and
        screen-space.
        """

        camera = self._camera
        scene_size = self._viewport.logical_size
        world_pos = self._object_to_control.world.position
        ndc_pos = la.vector_apply_matrix(world_pos, camera.camera_matrix)

        # Calculate direction pairs (world_directions, ndc_directions)
        base_directions = np.eye(3)
        if self._mode == "object":
            rot = self._object_to_control.local.rotation
            # The world direction is derived from the object
            world_directions = la.vector_apply_quaternion(
                base_directions, self._object_to_control.world.rotation
            )
            # Calculate ndc directions from here
            ndc_directions = (
                la.vector_apply_matrix(
                    world_pos + world_directions, camera.camera_matrix
                )
                - ndc_pos
            )
        elif self._mode == "world":
            # The world direction is the base direction
            rot = np.array((0, 0, 0, 1))  # null rotation
            world_directions = base_directions
            # Calculate ndc directions from here
            ndc_directions = (
                la.vector_apply_matrix(
                    world_pos + world_directions, camera.camera_matrix
                )
                - ndc_pos
            )
        elif self._mode == "screen":
            # The screen direction is the base_direction
            rot = camera.world.rotation
            screen_directions = base_directions * self._screen_size
            # Convert to world directions
            ndc_directions = screen_directions * (2, 2, -1) / (*scene_size, 1)
            world_directions = (
                la.vector_unproject(
                    (ndc_pos + ndc_directions)[:, :2], camera.projection_matrix
                )
                - world_pos
            )
        else:  # This cannot happen, in theory
            raise RuntimeError(f"Unexpected mode: `{self._mode}`")

        # Calculate screen directions from ndc_directions (also re-calculate for screen mode)
        # These represent how much one "step" moves on screen.
        # Note how for ndc_directions we have a valid z, but for screen_directions z is 0.
        screen_directions = ndc_directions * (*scene_size, 0) / (2, 2, 1)

        # Store direction lists, to be used during a drag operation.
        self._world_directions = world_directions
        self._ndc_directions = ndc_directions
        self._screen_directions = screen_directions

        # Apply rotation
        self.rotation = rot

    def _update_scale(
        self,
    ):
        """Update the scale of the gizmo so it gets the correct
        (approximate) size on screen.
        """

        # During interaction we don't adjust the scale, this way:
        # * During a rotation the gizmo does not flip,
        #   making it easier to see how much was rotated.
        # * During a translation the gizmo keeps its "world size",
        #   so that the perspective helps you see how the gizmo has moved.
        if self._ref:
            return

        camera = self._camera
        scene_size = self._viewport.logical_size
        world_pos = self._object_to_control.world.position
        ndc_pos = la.vector_apply_matrix(world_pos, camera.camera_matrix)

        # Get how our direction vectors express on screen
        ndc_sx = self._screen_size * 2 / scene_size[0]
        ndc_sy = self._screen_size * 2 / scene_size[1]
        vec1 = np.array((ndc_sx, 0))  # TODO: check how ndc_pos is consumed here
        vec1 = la.vector_unproject(vec1, camera.projection_matrix) - world_pos
        vec2 = np.array((0, ndc_sy))  # TODO: check how ndc_pos is consumed here
        vec2 = la.vector_unproject(vec2, camera.projection_matrix) - world_pos
        scale_scalar = 0.5 * (np.linalg.norm(vec1) + np.linalg.norm(vec2))

        # Determine the scale of this object, so that it gets the intended size on screen.
        scale = [scale_scalar, scale_scalar, scale_scalar]

        # Determine any flips so that the gizmo faces the camera. Note
        # that this checks whether the vector in question points away
        # from the camera.
        # -----------------------#  So on a view like this, where the widget
        #      | g               #  is on the left, the red leg might still
        # r ___|                 #  just point towards the camera, even though
        #     /                  #  it might not "feel" this way because the
        #    b                   #  blue leg may partly obscure elements of the
        # -----------------------#  red leg.
        for dim, vec in enumerate(self._ndc_directions):
            if vec[2] > 0:
                scale[dim] = -scale[dim]

        # Apply scale
        self.local.scale = np.array(scale)

    def _update_visibility(self):
        """Depending on the mode and the orientation of the camera,
        some elements are made invisible.
        """

        screen_directions = self._screen_directions

        # The scaled screen direction matches the size of the widget on screen.
        scale_scalar = abs(self.local.scale_x)
        scaled_screen_directions = scale_scalar * screen_directions

        # Determine what directions are orthogonal to the view plane
        show_direction = np.linalg.norm(scaled_screen_directions[:, :2], axis=-1) > 30

        # Also determine whether in-plane elements (arcs and translate2 handles) become hard to see
        vec1 = la.vector_normalize(screen_directions[[1, 2, 0], :])
        vec2 = la.vector_normalize(screen_directions[[2, 0, 1], :])
        show_direction2 = np.abs(np.sum(vec1 * vec2, axis=-1)) < 0.9

        if self._mode == "screen":
            # Show x and y lines and translate1 handles
            for dim, visible in enumerate([True, True, False]):
                self._line_children[dim].visible = visible
                self._translate1_children[dim].visible = visible
            # Show only the xy translate2 handle
            for dim, visible in enumerate([False, False, True]):
                self._translate2_children[dim].visible = visible
                self._arc_children[dim].visible = visible
        else:
            # Lines and arcs are always shown
            for dim in (0, 1, 2):
                self._arc_children[dim].visible = True
                self._line_children[dim].visible = True
            # The translate1 and scale handles depend on their angle to the camera
            for dim in (0, 1, 2):
                self._translate1_children[dim].visible = show_direction[dim]
                self._scale_children[dim].visible = show_direction[dim]
            # The translate2 handles depend on two angles to the camera
            for dim in (0, 1, 2):
                dim1, dim2 = [(1, 2), (2, 0), (0, 1)][dim]
                visible = (
                    show_direction[dim1]
                    and show_direction[dim2]
                    and show_direction2[dim]
                )
                self._translate2_children[dim].visible = visible

        # Per-dimension scaling is only possible in object-mode
        if self._mode != "object":
            for ob in self._scale_children[:3]:
                ob.visible = False

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

        # camera and/or viewport may have changed, may need to update internals
        if self._object_to_control is not None:
            self.update_gizmo(self._object_to_control.world)

        self.add_event_handler(
            self.process_event, "pointer_down", "pointer_move", "pointer_up", "wheel"
        )

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
            "event_pos": (event.x, event.y),
            "dim": ob.dim,
            "maxdist": 0,
            # Transform at time of start
            "scale": self._object_to_control.world.scale,
            "rot": self._object_to_control.world.rotation,
            "world_pos": ob_pos,
            "world_offset": ob_pos - this_pos,
            "ndc_pos": la.vector_apply_matrix(ob_pos, self._camera.projection_matrix),
            # Gizmo direction state at start-time of drag
            "flips": np.sign(self.world.scale),
            "world_directions": self._world_directions.copy(),
            "ndc_directions": self._ndc_directions.copy(),
            "screen_directions": self._screen_directions.copy(),
        }

    def _handle_translate_move(self, event):
        """Translate action, either using a translate1 or translate2 handle."""
        # Get dimensions to translate in
        dim = self._ref["dim"]

        # Get how the mouse has moved
        screen_moved = np.array(
            (
                event.x - self._ref["event_pos"][0],
                event.y - self._ref["event_pos"][1],
                0,
            )
        )
        scene_size = self._viewport.logical_size
        ndc_moved = screen_moved * np.array((2 / scene_size[0], -2 / scene_size[1], 0))

        # Init new position
        new_position = self._ref["world_pos"]
        new_position -= self._ref["world_offset"]

        if isinstance(dim, int):
            # For 1D movement we can project the screen movement to the world-direction.
            # Sample directions
            world_dir = self._ref["world_directions"][dim]
            ndc_dir = self._ref["ndc_directions"][dim]
            screen_dir = self._ref["screen_directions"][dim]
            # Calculate how many times the screen_dir matches the moved direction
            factor = get_scale_factor(screen_dir, screen_moved)
            # Calculate position by moving ndc_pos in that direction
            ndc_pos = self._ref["ndc_pos"] + factor * ndc_dir
            position = la.vector_unproject(ndc_pos[:2], self._camera.projection_matrix)
            # The found position has roundoff errors, let's align it with the world_dir
            world_move = position - self._ref["world_pos"]
            factor = get_scale_factor(world_dir, world_move)
            new_position += world_move
        else:
            # For 2d movement we project the cursor vector onto the plane defined
            # by the two world directions.
            dims = dim  # tuple of 2 ints
            # Get reference world pos and the world vectors. Imagine this a plane.
            world_pos = self._ref["world_pos"]
            world_dir1 = self._ref["world_directions"][dims[0]]
            world_dir2 = self._ref["world_directions"][dims[1]]
            # Get the line (in world coords) to move things to
            ndc_pos = self._ref["ndc_pos"]
            ndc_pos1 = ndc_pos + ndc_moved
            ndc_pos2 = ndc_pos1 + (0, 0, 1)
            cursor_world_pos1 = la.vector_unproject(
                ndc_pos1[:2], self._camera.projection_matrix
            )
            cursor_world_pos2 = la.vector_unproject(
                ndc_pos2[:2], self._camera.projection_matrix
            )
            # Get where line intersects plane, expressed in factors of the world dirs
            factor1, factor2 = get_line_plane_intersection(
                cursor_world_pos1, cursor_world_pos2, world_pos, world_dir1, world_dir2
            )
            new_position += factor1 * world_dir1
            new_position += factor2 * world_dir2

        # Apply
        self._object_to_control.local.position = new_position
        self.local.position = new_position

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
            npixels = get_scale_factor(ref_vec, screen_moved)
            scale = 2 ** (npixels / 100)
            scale = np.array((scale, scale, scale))
        else:
            # Get how much the mouse has moved in the ref direction
            screen_dir = self._ref["screen_directions"][dim]
            factor = get_scale_factor(screen_dir, screen_moved)
            factor *= self._ref["flips"][dim]
            npixels = factor * np.linalg.norm(screen_dir)
            # Calculate the relative scale
            scale = [1, 1, 1]
            scale[dim] = 2 ** (npixels / 100)
            scale = np.array(scale)

        # Apply
        self._object_to_control.local.scale = scale * self._ref["scale"]

    def _handle_rotate_move(self, event):
        """Rotate action."""
        # Get dimension around which to rotate, and the *other* dimensions
        dim = self._ref["dim"]
        dims = [(1, 2), (2, 0), (0, 1)][dim]

        # Get how the mouse has moved
        screen_moved = np.array(
            (
                event.x - self._ref["event_pos"][0],
                event.y - self._ref["event_pos"][1],
                0,
            )
        )

        # Calculate axis of rotation
        world_dir = self._ref["world_directions"][dim]
        axis = la.vector_normalize(world_dir)

        # Calculate the vector between the two arrows that span the arc.
        # We need to flip the sign in the right places to make this work.
        screen_dir1 = self._ref["screen_directions"][dims[0]]
        screen_dir2 = self._ref["screen_directions"][dims[1]]
        flip1, flip2 = self._ref["flips"][dims[0]], self._ref["flips"][dims[1]]
        screen_dir1 *= flip1
        screen_dir2 *= flip2
        screen_vec = screen_dir2 - screen_dir1

        # Now we can calculate how far the mouse moved in *that* direction.
        factor = get_scale_factor(screen_vec, screen_moved)
        factor = factor * flip1 * flip2
        angle = factor * 2

        # Apply
        rot = la.quaternion_make_from_axis_angle(axis, angle)
        self._object_to_control.local.rotation = la.quaternion_multiply(
            rot, self._ref["rot"]
        )


def get_scale_factor(vec1, vec2):
    """Calculate how many times vec2 fits onto vec1. Basically a dot
    product and a division by their norms.
    """
    factor = np.dot(la.vector_normalize(vec2), la.vector_normalize(vec1))
    factor *= np.linalg.norm(vec2) / np.linalg.norm(vec1)
    return factor


def get_line_plane_intersection(a0, a1, p0, v1, v2):
    """Get the intersection point of a line onto a plane.
    Args a0 and a1 represent two points on a line. Arg p0 is a point
    on a plane, and v1 and v2 two in-plane vectors. The intersection
    is expressed in factors of v1 and v2.
    """
    # Get the vector from a0 to a1
    av = a1 - a0

    # Get how often this vector must be applied from a0 to get to the plane
    # https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
    n = np.cross(la.vector_normalize(v1), la.vector_normalize(v2))
    at = np.dot(p0 - a0, n) / np.dot(av, n)

    # So the point where the line intersects the plane is ...
    p1 = a0 + at * av

    # But let's re-express that in a factor of v1 and v2, so that
    # we really only move in these directions.
    v3 = p1 - p0
    return get_scale_factor(v1, v3), get_scale_factor(v2, v3)
