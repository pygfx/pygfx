"""
A transform gizmo to manipulate world objects.
"""


import numpy as np
import pygfx as gfx
from ..objects import WorldObject
from pygfx.linalg import Vector3


class TransformGizmo(WorldObject):
    """
    A gizmo object that can be used to manipulate (i.e. transform) a world object.

    Aguments:
        object_to_control (WorldObject): the object to transform with the gizmo.
        screen_size (float): the approximate size of the widget in logical pixels.
    """

    def __init__(self, object_to_control, screen_size=100):
        super().__init__()
        self._create_components()
        self.set_object(object_to_control)

        self._renderer = None
        self._canvas = None
        self._camera = None

        # The (approximate) size of the gizmo on screen
        self._screen_size = float(screen_size)

        # The mode of operation
        self.toggle_mode("object")  # object, world, screen

        # A dict that is a reference for the pointer down event. Or None.
        self._ref = None

    def set_object(self, object_to_control):
        """Set the WorldObject to control with the gizmo."""
        assert isinstance(object_to_control, WorldObject)
        self._object_to_control = object_to_control

    def toggle_mode(self, mode=None):
        modes = "object", "world", "screen"
        if not mode:
            mode = {"object": "world", "world": "screen"}.get(self._mode, "object")
        if mode not in modes:
            raise ValueError(f"Invalid mode '{mode}', must be one of {modes}.")
        print("mode is now", mode)
        self._mode = mode
        self._on_mode_switch()

    def add_default_event_handlers(self, renderer, camera):
        # todo: update other methods with the same name to include renderer?
        canvas = renderer.target
        self._renderer = renderer
        self._canvas = canvas
        self._camera = camera
        canvas.add_event_handler(
            self._handle_event,
            "pointer_down",
            "pointer_move",
            "pointer_up",
        )

    def _create_components(self):

        self._halfway = halfway = 0.65

        # The lines and arcs are not apaque. That way they're also not
        # pickable, so they won't be "in the way".
        line_opacity = 0.6

        # --- the parts that are at the center

        # Create screen translate handle
        sphere_geo = gfx.sphere_geometry(0.07)
        scale_uniform = gfx.Mesh(
            sphere_geo,
            gfx.MeshBasicMaterial(color="#ffffff"),
        )
        scale_uniform.scale.set(1.5, 1.5, 1.5)
        scale_uniform.dim = None

        # --- the parts that are fully in one dimension

        # Create lines
        line_geo = gfx.Geometry(positions=[(0, 0, 0), (halfway, 0, 0)])
        line_x = gfx.Line(
            line_geo,
            gfx.LineMaterial(thickness=4, color="#ff0000", opacity=line_opacity),
        )
        line_y = gfx.Line(
            line_geo,
            gfx.LineMaterial(thickness=4, color="#00ff00", opacity=line_opacity),
        )
        line_z = gfx.Line(
            line_geo,
            gfx.LineMaterial(thickness=4, color="#0000ff", opacity=line_opacity),
        )

        # Create 1D translate handles
        cone_geo = gfx.cone_geometry(0.1, 0.17)
        cone_geo.positions.data[:] = cone_geo.positions.data[:, ::-1]
        translate_x = gfx.Mesh(
            cone_geo,
            gfx.MeshBasicMaterial(color="#ff0000"),
        )
        translate_y = gfx.Mesh(
            cone_geo,
            gfx.MeshBasicMaterial(color="#00ff00"),
        )
        translate_z = gfx.Mesh(
            cone_geo,
            gfx.MeshBasicMaterial(color="#0000ff"),
        )
        translate_x.position.set(1, 0, 0)
        translate_y.position.set(0, 1, 0)
        translate_z.position.set(0, 0, 1)

        # Create scale handles
        cube_geo = gfx.box_geometry(0.1, 0.1, 0.1)
        scale_x = gfx.Mesh(
            cube_geo,
            gfx.MeshBasicMaterial(color="#ff0000"),
        )
        scale_y = gfx.Mesh(
            cube_geo,
            gfx.MeshBasicMaterial(color="#00ff00"),
        )
        scale_z = gfx.Mesh(
            cube_geo,
            gfx.MeshBasicMaterial(color="#0000ff"),
        )
        scale_x.position.set(halfway, 0, 0)
        scale_y.position.set(0, halfway, 0)
        scale_z.position.set(0, 0, halfway)

        # --- the parts that are in a plane

        # Create arcs
        t = np.linspace(0, np.pi / 2, 64)
        arc_positions = np.stack([0 * t, np.sin(t), np.cos(t)], 1).astype(np.float32)
        arc_geo = gfx.Geometry(positions=arc_positions * halfway)
        arc_yz = gfx.Line(
            arc_geo,
            gfx.LineMaterial(thickness=4, color="#880000", opacity=line_opacity),
        )
        arc_zx = gfx.Line(
            arc_geo,
            gfx.LineMaterial(thickness=4, color="#008800", opacity=line_opacity),
        )
        arc_xy = gfx.Line(
            arc_geo,
            gfx.LineMaterial(thickness=4, color="#000088", opacity=line_opacity),
        )
        arc_zx.scale.y = -1
        arc_xy.scale.z = -1

        # Create in-plane translate handles
        plane_geo = gfx.box_geometry(0.01, 0.15, 0.15)
        translate_yz = gfx.Mesh(
            plane_geo,
            gfx.MeshBasicMaterial(color="#ff0000"),
        )
        translate_zx = gfx.Mesh(
            plane_geo,
            gfx.MeshBasicMaterial(color="#00ff00"),
        )
        translate_xy = gfx.Mesh(
            plane_geo,
            gfx.MeshBasicMaterial(color="#0000ff"),
        )
        inside_arc = 0.4 * halfway
        translate_yz.position.set(0, inside_arc, inside_arc)
        translate_zx.position.set(inside_arc, 0, inside_arc)
        translate_xy.position.set(inside_arc, inside_arc, 0)

        # Create rotation handles
        # These are positioned on each mode switch
        rotate_yz = gfx.Mesh(
            sphere_geo,
            gfx.MeshBasicMaterial(color="#ff0000"),
        )
        rotate_zx = gfx.Mesh(
            sphere_geo,
            gfx.MeshBasicMaterial(color="#00ff00"),
        )
        rotate_xy = gfx.Mesh(
            sphere_geo,
            gfx.MeshBasicMaterial(color="#0000ff"),
        )

        # --- post-process

        # Rotate objects to their correct orientation
        for ob in [line_y, translate_y, scale_y, arc_zx, translate_zx, rotate_zx]:
            ob.rotation.set_from_axis_angle(Vector3(0, 0, 1), np.pi / 2)
        for ob in [line_z, translate_z, scale_z, arc_xy, translate_xy, rotate_xy]:
            ob.rotation.set_from_axis_angle(Vector3(0, -1, 0), np.pi / 2)

        # Store the objectss
        self._center_sphere = scale_uniform
        self._line_children = line_x, line_y, line_z
        self._arc_children = arc_yz, arc_zx, arc_xy
        self._translate1_children = translate_x, translate_y, translate_z
        self._translate2_children = translate_yz, translate_zx, translate_xy
        self._translate_children = self._translate1_children + self._translate2_children
        self._scale_children = scale_x, scale_y, scale_z
        self._rotate_children = rotate_yz, rotate_zx, rotate_xy

        # Assign dimensions
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

        # Update position of rotation handles
        rotate_yz, rotate_zx, rotate_xy = self._rotate_children
        halfway = self._halfway
        on_arc = halfway * 2**0.5 / 2
        if self._mode == "screen":
            rotate_yz.position.set(0, on_arc, 0)
            rotate_zx.position.set(on_arc, 0, 0)
            rotate_xy.position.set(on_arc, on_arc, 0)
        else:
            rotate_yz.position.set(0, on_arc, on_arc)
            rotate_zx.position.set(on_arc, 0, on_arc)
            rotate_xy.position.set(on_arc, on_arc, 0)

    def update_matrix_world(self, *args, **kwargs):
        # This gets called by the renderer just before rendering.
        self._adjust_to_camera()
        super().update_matrix_world(*args, **kwargs)

    def _adjust_to_camera(self):
        if not (self._object_to_control and self._canvas and self._camera):
            return
        camera = self._camera
        canvas_size = self._canvas.get_logical_size()

        # Get center position (of the gizmo) in world and screen coords
        world_pos = self._object_to_control.position
        ndc_pos = world_pos.clone().project(camera)

        # Get how our direction vectors express on screen
        ndc_size_x = self._screen_size * 2 / canvas_size[0]
        ndc_size_y = self._screen_size * 2 / canvas_size[1]
        vec1 = (
            ndc_pos.clone()
            .add(Vector3(ndc_size_x, 0, 0))
            .unproject(camera)
            .sub(world_pos)
        )
        vec2 = (
            ndc_pos.clone()
            .add(Vector3(0, ndc_size_y, 0))
            .unproject(camera)
            .sub(world_pos)
        )

        # Determine the scale of this object, so that it gets the intended size on screen.
        # We may flip the scale for some dimensions further down.
        scale_scalar = 0.5 * (vec1.length() + vec2.length())
        scale = [scale_scalar, scale_scalar, scale_scalar]

        # Calculate direction pairs (world_directions, ndc_directions)
        base_directions = Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)
        assert self._mode in ("object", "world", "screen")

        if self._mode == "object":
            # The world direction is derived from the object
            rot = self._object_to_control.rotation.clone()
            world_directions = [vec.apply_quaternion(rot) for vec in base_directions]
            # Calculate ndc directions from here
            ndc_directions = [
                world_pos.clone().add(vec).project(camera).sub(ndc_pos)
                for vec in world_directions
            ]
        elif self._mode == "world":
            # The world direction is the base direction
            rot = gfx.linalg.Quaternion()  # null rotation
            world_directions = base_directions
            # Calculate ndc directions from here
            ndc_directions = [
                world_pos.clone().add(vec).project(camera).sub(ndc_pos)
                for vec in world_directions
            ]
        elif self._mode == "screen":
            # The screen direction is the base_direction
            rot = gfx.linalg.Quaternion().set_from_rotation_matrix(camera.matrix_world)
            screen_directions = [
                vec.multiply_scalar(self._screen_size) for vec in base_directions
            ]
            # Convert to world directions
            ndc_directions = [
                Vector3(vec.x / canvas_size[0] * 2, vec.y / canvas_size[1] * 2, -vec.z)
                for vec in screen_directions
            ]
            world_directions = [
                ndc_pos.clone().add(vec).unproject(camera).sub(world_pos)
                for vec in ndc_directions
            ]

        # Calculate screen directions from ndc_directions. Also recalculate for screen mode.
        # These represent how much one "step" moves on screen.
        screen_directions = [
            Vector3(vec.x * canvas_size[0] / 2, -vec.y * canvas_size[1] / 2, 0)
            for vec in ndc_directions
        ]

        # The scaled version matches the size of the widget on screen.
        scaled_screen_directions = [
            vec.clone().multiply_scalar(scale_scalar) for vec in screen_directions
        ]

        # Store direction lists
        self._world_directions = world_directions
        self._ndc_directions = ndc_directions
        self._screen_directions = screen_directions

        # Determine any flips so that the gizmo faces the camera. Note
        # that this checks whether the vector in question points away
        # from the camera.
        # -----------------------#  So on a canvas like this, where the widget
        #      | g              #  is on the left, the red leg might still
        # r ___|                #  just point towards the camera, even though
        #     /                 #  it might not "feel" this way because the
        #    b                  #  blue leg may partly obscure elements of the
        # -----------------------#  red leg.
        for dim, vec in enumerate(ndc_directions):
            if vec.z > 0:
                scale[dim] = -scale[dim]

        self.scale = Vector3(*scale)
        self.rotation = rot

        # -- Hide objects for which the direction (on screen) becomes ill-defined.

        # Determine what directions are orthogonal to the view plane
        show_direction = [True, True, True]
        for dim, vec in enumerate(scaled_screen_directions):
            size = (vec.x**2 + vec.y**2) ** 0.5
            show_direction[dim] = size > 30  # in pixels

        # Also get whether in-plane elements become hard to see
        show_direction2 = [True, True, True]
        for dim, vec in enumerate(screen_directions):
            dims = [(1, 2), (2, 0), (0, 1)][dim]
            vec1 = screen_directions[dims[0]].clone().normalize()
            vec2 = screen_directions[dims[1]].clone().normalize()
            show_direction2[dim] = abs(vec1.dot(vec2)) < 0.9

        if self._mode == "screen":
            for dim, visible in enumerate([True, True, False]):
                self._line_children[dim].visible = visible
                self._translate1_children[dim].visible = visible
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

    def _handle_event(self, event):
        # todo: check buttons and modifiers
        type = event["event_type"]
        if type in "pointer_down":
            if event["modifiers"]:
                return
            self._ref = None
            # todo: make renderer cache pick info calls for the same frame
            info = self._renderer.get_pick_info((event["x"], event["y"]))
            ob = info["world_object"]
            if ob == self._center_sphere:
                self._handle_start("scale", event, ob)
            elif ob in self._translate_children:
                self._handle_start("translate", event, ob)
            elif ob in self._scale_children:
                self._handle_start("scale", event, ob)
            elif ob in self._rotate_children:
                self._handle_start("rotate", event, ob)
        elif type == "pointer_up":
            if self._ref:
                if self._ref["dim"] is None and self._ref["maxdist"] < 3:
                    self.toggle_mode()
                self._ref = None
        elif type == "pointer_move":
            if not self._ref:
                return
            dist = (
                (event["x"] - self._ref["event"]["x"]) ** 2
                + (event["y"] - self._ref["event"]["y"]) ** 2
            ) ** 0.5
            self._ref["maxdist"] = max(self._ref["maxdist"], dist)
            if self._ref["maxdist"] < 3:
                pass
            elif self._ref["kind"] == "translate":
                self._handle_translate_move(event)
            elif self._ref["kind"] == "scale":
                self._handle_scale_move(event)
            elif self._ref["kind"] == "rotate":
                self._handle_rotate_move(event)

    def _handle_start(self, kind, event, ob):
        sign = np.sign
        this_pos = self._object_to_control.position.clone()
        ob_pos = ob.get_world_position().clone()
        self._ref = {
            "kind": kind,
            "event": event,
            "dim": ob.dim,
            "maxdist": 0,
            # Transform at time of start
            "scale": self._object_to_control.scale.clone(),
            "rot": self._object_to_control.rotation.clone(),
            "world_pos": ob_pos,
            "world_offset": ob_pos.clone().sub(this_pos),
            "ndc_pos": ob_pos.clone().project(self._camera),
            # Gizmo direction state at time of start
            "flips": [sign(self.scale.x), sign(self.scale.y), sign(self.scale.z)],
            "world_directions": [vec.clone() for vec in self._world_directions],
            "ndc_directions": [vec.clone() for vec in self._ndc_directions],
            "screen_directions": [vec.clone() for vec in self._screen_directions],
        }

    def _handle_translate_move(self, event):
        # Get dimensions to translate in
        dim = self._ref["dim"]

        # Get how the mouse has moved
        screen_moved = Vector3(
            event["x"] - self._ref["event"]["x"],
            event["y"] - self._ref["event"]["y"],
            0,
        )
        canvas_size = self._canvas.get_logical_size()
        ndc_moved = screen_moved.clone().multiply(
            Vector3(2 / canvas_size[0], -2 / canvas_size[1], 0)
        )

        # Init new position
        new_position = self._ref["world_pos"].clone()
        new_position.sub(self._ref["world_offset"])

        if isinstance(dim, int):
            # For 1D movement we can project the screen movement to the world-direction.
            # Sample directions
            world_dir = self._ref["world_directions"][dim]
            ndc_dir = self._ref["ndc_directions"][dim].clone()
            screen_dir = self._ref["screen_directions"][dim].clone()
            # Calculate how many times the screen_dir matches the moved direction
            factor = get_scale_factor(screen_dir, screen_moved)
            # Calculate position by moving ndc_pos in that direction
            ndc_pos = self._ref["ndc_pos"].clone().add_scaled_vector(ndc_dir, factor)
            position = ndc_pos.unproject(self._camera)
            # The found position has roundoff errors, let's align it with the world_dir
            world_move = position.clone().sub(self._ref["world_pos"])
            factor = get_scale_factor(world_dir, world_move)
            new_position.add_scaled_vector(world_move, 1)  # world_dir, factor)
        else:
            # For 2d movement we project the cursor vector onto the plane defined
            # by the two world directions.
            dims = dim  # tuple of 2 ints
            # Get reference world pos and the world vectors. Imagine this a plane.
            world_pos = self._ref["world_pos"].clone()
            world_dir1 = self._ref["world_directions"][dims[0]]
            world_dir2 = self._ref["world_directions"][dims[1]]
            # Get the line (in world coords) to move things to
            ndc_pos = self._ref["ndc_pos"].clone()
            ndc_pos1 = ndc_pos.add_scaled_vector(ndc_moved, 1)
            ndc_pos2 = ndc_pos1.clone().add(Vector3(0, 0, 1))
            cursor_world_pos1 = ndc_pos1.unproject(self._camera)
            cursor_world_pos2 = ndc_pos2.unproject(self._camera)
            # Get where line intersects plane, expressed in factors of the world dirs
            factor1, factor2 = get_line_plane_intersection(
                cursor_world_pos1, cursor_world_pos2, world_pos, world_dir1, world_dir2
            )
            new_position.add_scaled_vector(world_dir1, factor1)
            new_position.add_scaled_vector(world_dir2, factor2)

        # Apply
        self._object_to_control.position = new_position.clone()
        self.position = new_position.clone()

    def _handle_scale_move(self, event):

        # Get dimension
        dim = self._ref["dim"]

        # Get how the mouse has moved
        screen_moved = Vector3(
            event["x"] - self._ref["event"]["x"],
            event["y"] - self._ref["event"]["y"],
            0,
        )

        if dim is None:
            # Uniform scale
            ref_vec = Vector3(1, -1, 0)
            npixels = get_scale_factor(ref_vec, screen_moved)
            scale = 2 ** (npixels / 100)
            scale = Vector3(scale, scale, scale)
        else:
            # Get how much the mouse has moved in the ref direction
            screen_dir = self._ref["screen_directions"][dim]
            factor = get_scale_factor(screen_dir, screen_moved)
            factor *= self._ref["flips"][dim]
            npixels = factor * screen_dir.length()
            # Calculate the relative scale
            scale = [1, 1, 1]
            scale[dim] = 2 ** (npixels / 100)
            scale = Vector3(*scale)

        # Apply
        self._object_to_control.scale = self._ref["scale"].clone().multiply(scale)

    def _handle_rotate_move(self, event):
        # Get dimension around which to rotate, and the *other* dimensions
        dim = self._ref["dim"]
        dims = [(1, 2), (2, 0), (0, 1)][dim]

        # Get how the mouse has moved
        screen_moved = Vector3(
            event["x"] - self._ref["event"]["x"],
            event["y"] - self._ref["event"]["y"],
            0,
        )

        # Calculate axis of rotation
        world_dir = self._ref["world_directions"][dim]
        axis = world_dir.clone().normalize()

        # Calculate the vector between the two arrows than span the arc.
        # We need to flip the sign in the right places to make this work.
        screen_dir1 = self._ref["screen_directions"][dims[0]].clone()
        screen_dir2 = self._ref["screen_directions"][dims[1]].clone()
        flip1, flip2 = self._ref["flips"][dims[0]], self._ref["flips"][dims[1]]
        screen_dir1.multiply_scalar(flip1)
        screen_dir2.multiply_scalar(flip2)
        screen_vec = screen_dir2.sub(screen_dir1)

        # Now we can calculate how far the mouse moved in *that* direction.
        factor = get_scale_factor(screen_vec, screen_moved)
        factor = factor * flip1 * flip2
        angle = factor * 2

        # Apply
        rot = gfx.linalg.Quaternion().set_from_axis_angle(axis, angle)
        self._object_to_control.rotation = rot.multiply(self._ref["rot"])


def get_scale_factor(vec1, vec2):
    """Calculate how many times vec2 fits onto vec1. Basically a dot
    product and a division by their norms.
    """
    factor = vec2.clone().normalize().dot(vec1.clone().normalize())
    factor *= vec2.length() / vec1.length()
    return factor


def get_line_plane_intersection(a0, a1, p0, v1, v2):
    """Get the intersection point of a line onto a plane.
    Args a0 and a1 represent two points on a line. Arg p0 is a point
    on a plane, and v1 and v2 two in-plane vectors. The interserction
    is expressed in factors of v1 and v2.
    """
    # Get the vector from a0 to a1
    av = a1.clone().sub(a0)

    # Get how often this vector must be applied from a0 to get to the plane
    # https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
    n = v1.clone().normalize().cross(v2.clone().normalize())
    nom = p0.clone().sub(a0).dot(n)
    denom = av.clone().dot(n)
    at = nom / denom

    # So the point where the line intersects the plane is ...
    p1 = a0.clone().add_scaled_vector(av, at)
    v3 = p1.clone().sub(p0)

    return get_scale_factor(v1, v3), get_scale_factor(v2, v3)
