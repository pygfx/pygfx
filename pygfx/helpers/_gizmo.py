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
        self._mode = "object"  # object, world, screen

        # A dict that is a reference for the pointer down event. Or None.
        self._ref = None

        # Two vectors in world coords, representing screen-x and screen-y
        # todo: can remove?
        self._screen_vecs = Vector3(0, 0, 0), Vector3(0, 0, 0)

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

        halfway = 0.65

        # --- the parts that are at the center

        # Create screen translate handle
        sphere_geo = gfx.sphere_geometry(0.07)
        translate_screen = gfx.Mesh(
            sphere_geo,
            gfx.MeshBasicMaterial(color="#ffffff"),
        )
        translate_screen.scale.set(1.5, 1.5, 1.5)

        # --- the parts that are fully in one dimension

        # Create lines
        line_geo = gfx.Geometry(positions=[(0, 0, 0), (1, 0, 0)])
        line_x = gfx.Line(
            line_geo,
            gfx.LineMaterial(thickness=4, color="#880000"),
        )
        line_y = gfx.Line(
            line_geo,
            gfx.LineMaterial(thickness=4, color="#008800"),
        )
        line_z = gfx.Line(
            line_geo,
            gfx.LineMaterial(thickness=4, color="#000088"),
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
            gfx.LineMaterial(thickness=4, color="#880000"),
        )
        arc_zx = gfx.Line(
            arc_geo,
            gfx.LineMaterial(thickness=4, color="#008800"),
        )
        arc_xy = gfx.Line(
            arc_geo,
            gfx.LineMaterial(thickness=4, color="#000088"),
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
        on_arc = halfway * 2**0.5 / 2
        rotate_yz.position.set(0, on_arc, on_arc)
        rotate_zx.position.set(on_arc, 0, on_arc)
        rotate_xy.position.set(on_arc, on_arc, 0)

        # --- post-process

        # Rotate objects to their correct orientation
        for ob in [line_y, translate_y, scale_y, arc_zx, translate_zx, rotate_zx]:
            ob.rotation.set_from_axis_angle(Vector3(0, 0, 1), np.pi / 2)
        for ob in [line_z, translate_z, scale_z, arc_xy, translate_xy, rotate_xy]:
            ob.rotation.set_from_axis_angle(Vector3(0, -1, 0), np.pi / 2)

        # Store the objectss
        self._line_children = line_x, line_y, line_z
        self._arc_children = arc_yz, arc_zx, arc_xy
        self._translate0_children = (translate_screen,)
        self._translate1_children = translate_x, translate_y, translate_z
        self._translate2_children = translate_yz, translate_zx, translate_xy
        self._translate_children = (
            self._translate0_children
            + self._translate1_children
            + self._translate2_children
        )
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
        translate_screen.dim = "screen"
        for i, ob in enumerate(self._translate2_children):
            dim = [0, 1, 2]
            dim.pop(i)
            ob.dim = tuple(dim)

        # Attach to the gizmo object
        self.add(*self._line_children)
        self.add(*self._arc_children)
        self.add(*self._translate_children)
        self.add(*self._scale_children)
        self.add(*self._rotate_children)

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
        center_world = self._object_to_control.position
        center_screen = center_world.clone().project(camera)

        # Get how our direction vectors express on screen
        pos1 = center_screen.clone().add(Vector3(100, 0, 0)).unproject(camera)
        pos2 = center_screen.clone().add(Vector3(0, 100, 0)).unproject(camera)
        pos1.multiply_scalar(2 / 100 / canvas_size[0])
        pos2.multiply_scalar(2 / 100 / canvas_size[1])
        self._screen_vecs = pos1, pos2

        # Determine the scale of this object
        avg_length = 0.5 * (
            self._screen_vecs[0].length() + self._screen_vecs[1].length()
        )
        scale_scalar = self._screen_size * avg_length
        scale = [scale_scalar, scale_scalar, scale_scalar]

        # Calculate direction pairs (world, screen)
        base_directions = Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)
        if self._mode == "object":
            # The world direction is derived from the object
            rot = self._object_to_control.rotation.clone()
            world_directions = [vec.apply_quaternion(rot) for vec in base_directions]
            # Convert to screen directions
            ndc_directions = [
                center_world.clone().add(vec).project(camera).sub(center_screen)
                # .multiply_scalar(scale_scalar)
                for vec in world_directions
            ]

        elif self._mode == "world":
            # The world direction is the base direction
            rot = gfx.linalg.Quaternion()
            world_directions = base_directions
            # Convert to screen directions
            ndc_directions = [
                center_world.clone().add(vec).project(camera).sub(center_screen)
                # .multiply_scalar(scale_scalar)
                for vec in world_directions
            ]

        else:
            # The screen direction is the base_direction
            # TODO: perspective projection can show the other axis of the gizmo, what does this mean?
            # cam_unproject = camera.projection_matrix_inverse.clone().multiply(camera.matrix_world)
            # cam_unproject = camera.matrix_world.clone().multiply(camera.projection_matrix_inverse)
            # cam_project = camera.projection_matrix.clone().multiply(camera.matrix_world_inverse)
            rot = gfx.linalg.Quaternion().set_from_rotation_matrix(
                camera.matrix_world
            )  # .inverse()
            screen_directions = [
                vec.multiply_scalar(self._screen_size) for vec in base_directions
            ]
            # Convert to world directions
            ndc_directions = [
                Vector3(vec.x / canvas_size[0] * 2, vec.y / canvas_size[1] * 2, vec.z)
                for vec in screen_directions
            ]
            world_directions = [
                center_screen.clone().add(vec).unproject(camera).sub(center_world)
                for vec in ndc_directions
            ]

        screen_directions = [
            Vector3(vec.x * canvas_size[0] / 2, -vec.y * canvas_size[1] / 2, 0)
            for vec in ndc_directions
        ]
        # Store
        self._world_directions = world_directions
        self._ndc_directions = ndc_directions
        self._screen_directions = screen_directions

        # Determine what directions are orthogonal to the view plane
        # And also a multiplier to compensate for the smaller leverage at a high angle
        show_direction = [True, True, True]
        multipliers = [1, 1, 1]
        for dim, vec in enumerate(screen_directions):
            size = (vec.x**2 + vec.y**2) ** 0.5
            show_direction[dim] = size > 15  # in pixels
            multipliers[dim] = 1 / (size + 0.1)

        # Hide plane-translation handles if ill-defined
        for dim in (0, 1, 2):
            dims = [(1, 2), (2, 0), (0, 1)][dim]
            vec1 = screen_directions[dims[0]].clone().normalize()
            vec2 = screen_directions[dims[1]].clone().normalize()
            in_plane_measure = abs(vec1.dot(vec2))
            self._translate2_children[dim].visible = in_plane_measure < 0.75

        # Store multipler per direction
        ref_multiplier = min(multipliers)
        self._direction_multipliers = [x / ref_multiplier for x in multipliers]

        # Hide object for which the direction (on screen) becomes ill-defined.
        for dim in (0, 1, 2):
            self._line_children[dim].visible = show_direction[dim]
            self._translate1_children[dim].visible = show_direction[dim]
            self._scale_children[dim].visible = show_direction[dim]

        # Determine any flips so that the gizmo faces the camera
        for dim, vec in enumerate(ndc_directions):
            if vec.z > 0:
                scale[dim] = -scale[dim]
                # self._screen_directions[dim].multiply_scalar(-1)
                # dir_screen = self._screen_directions[dim]
                # self._screen_directions[dim].multiply_scalar(-1)# = Vector3(-dir_screen.x, -dir_screen.y, -dir_screen.z)
                # self._world_directions[dim].multiply_scalar(-1)

        self.scale = Vector3(*scale)
        self.rotation = rot

        self._ndc_pos = self.position.clone().project(camera)
        self._screen_pos = Vector3(
            (self._ndc_pos.x + 1) * canvas_size[0] / 2,
            (self._ndc_pos.y + 1) * canvas_size[1] / 2,
            0,
        )

    # %% Utils

    def _get_world_vector_from_pointer_move(self, event):
        dx = event["x"] - self._ref["event"]["x"]
        dy = event["y"] - self._ref["event"]["y"]
        screen_vecs = self._ref["screen_vecs"]  # the self._screen_vecs changes
        return (
            screen_vecs[0]
            .clone()
            .multiply_scalar(-dx)
            .add_scaled_vector(screen_vecs[1], +dy)
        )

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
            if ob in self._translate0_children:
                return self.toggle_mode()
            elif ob in self._translate_children:
                self._handle_translate_start(event, ob)
            elif ob in self._scale_children:
                self._handle_scale_start(event, ob)
            elif ob in self._rotate_children:
                self._handle_rotate_start(event, ob)
        elif type == "pointer_up":
            self._ref = None
        elif type == "pointer_move":
            if not self._ref:
                pass
            elif self._ref["kind"] == "translate":
                self._handle_translate_move(event)
            elif self._ref["kind"] == "scale":
                self._handle_scale_move(event)
            elif self._ref["kind"] == "rotate":
                self._handle_rotate_move(event)

    def _handle_translate_start(self, event, ob):
        self._ref = {
            "kind": "translate",
            "event": event,
            "world_pos": self._object_to_control.position.clone(),
            "ndc_pos": self._ndc_pos.clone(),
            "dim": ob.dim,
            "world_directions": [vec.clone() for vec in self._world_directions],
            "ndc_directions": [vec.clone() for vec in self._ndc_directions],
            "screen_directions": [vec.clone() for vec in self._screen_directions],
        }

    def _handle_translate_move(self, event):
        # Get dimensions to translate in
        dim = self._ref["dim"]
        if isinstance(dim, int):
            dims = [dim]
        else:  # tuple
            dims = dim

        # Get how the mouse has moved
        screen_moved = Vector3(
            event["x"] - self._ref["event"]["x"],
            event["y"] - self._ref["event"]["y"],
            0,
        )

        # Init new position
        new_position = self._ref["world_pos"].clone()

        for dim in dims:
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
            new_position.add_scaled_vector(world_dir, factor)

        # Apply
        self._object_to_control.position = new_position.clone()
        self.position = new_position.clone()

    def _handle_scale_start(self, event, ob):
        self._ref = {
            "kind": "scale",
            "event": event,
            "scale": self._object_to_control.scale.clone(),
            "screen_vecs": self._screen_vecs,
            "dim": ob.dim,
        }

    def _handle_scale_move(self, event):
        dim = self._ref["dim"]
        # Calculate how far the mouse has moved
        dx = event["x"] - self._ref["event"]["x"]
        dy = event["y"] - self._ref["event"]["y"]
        dir_x, dir_y = self._screen_directions[dim]
        dir_norm = (dir_x**2 + dir_y**2) ** 0.5
        dist_pixels = dir_x * dx / dir_norm - dir_y * dy / dir_norm
        # Turn that into a scale vector.
        # We can only scale in object coordinates - there is no way to
        # "rotate a scale transform"
        scale = [1, 1, 1]
        scale[dim] = 2 ** (dist_pixels / 100)
        scale = Vector3(*scale)
        # Apply
        self._object_to_control.scale = self._ref["scale"].clone().multiply(scale)

    def _handle_rotate_start(self, event, ob):
        self._ref = {
            "kind": "rotate",
            "event": event,
            "rot": self._object_to_control.rotation.clone(),
            "dim": ob.dim,
            "world_directions": [vec.clone() for vec in self._world_directions],
            "ndc_directions": [vec.clone() for vec in self._ndc_directions],
            "screen_directions": [vec.clone() for vec in self._screen_directions],
        }

    def _handle_rotate_move(self, event):
        dim = self._ref["dim"]
        # dims = [(1, 2), (2, 0), (0, 1)][dim]
        # Calculate axis of rotation
        axis = self._ref["world_directions"][dim].clone().normalize()
        # Calculate how far the mouse has moved
        dx = event["x"] - self._ref["event"]["x"]
        dy = event["y"] - self._ref["event"]["y"]
        #
        # screen_directions = self._ref["screen_directions"]
        # vec1, vec2 = screen_directions[dims[0]], screen_directions[dims[1]]
        # vec12 = vec2.clone().sub(vec1).normalize()
        # dist_pixels = vec12.x * dx + vec12.y * dy

        dir_x, dir_y, _ = self._ref["screen_directions"][dim].to_array()
        dir_norm = (dir_x**2 + dir_y**2) ** 0.5
        if dir_norm == 0:
            dir_norm = 1
            dir_x = dir_y = -0.7071
        dist_pixels = dir_x * dy / dir_norm + dir_y * dx / dir_norm
        # Calculate angle
        angle = dist_pixels / 50
        # Apply
        rot = gfx.linalg.Quaternion().set_from_axis_angle(axis, angle)
        self._object_to_control.rotation = rot.multiply(self._ref["rot"])


# def world_vec_to_screen_vec(center_world, center_screen, world_directions, camera,)


def get_scale_factor(vec1, vec2):
    factor = vec2.clone().normalize().dot(vec1.clone().normalize())
    factor *= vec2.length() / vec1.length()
    return factor
