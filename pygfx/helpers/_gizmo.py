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

        # A dict that is a reference for the pointer down event. Or None.
        self._ref = None

        # Two vectors in world coords, representing screen-x and screen-y
        self._screen_vecs = Vector3(0, 0, 0), Vector3(0, 0, 0)

    def set_object(self, object_to_control):
        """Set the WorldObject to control with the gizmo."""
        assert isinstance(object_to_control, WorldObject)
        self._object_to_control = object_to_control

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

        # Create translate-screen handle
        sphere_geo = gfx.sphere_geometry(0.07)
        translate_screen = gfx.Mesh(
            sphere_geo,
            gfx.MeshBasicMaterial(color="#ffffff"),
        )
        translate_screen.scale.set(1.5, 1.5, 1.5)

        # Create translate handles
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

        # ---

        # Rotate objects to their correct orientation
        for ob in [line_y, translate_y, scale_y, arc_zx]:
            ob.rotation.set_from_axis_angle(Vector3(0, 0, 1), np.pi / 2)
        for ob in [line_z, translate_z, scale_z, arc_xy]:
            ob.rotation.set_from_axis_angle(Vector3(0, -1, 0), np.pi / 2)

        # Store the objectss
        self._line_children = line_x, line_y, line_z
        self._arc_children = arc_yz, arc_zx, arc_xy
        self._translate_children = (
            translate_x,
            translate_y,
            translate_z,
            translate_screen,
        )
        self._scale_children = scale_x, scale_y, scale_z
        self._rotate_children = rotate_yz, rotate_zx, rotate_xy

        # Assign dimensions
        for triplet in [
            self._line_children,
            self._translate_children[:3],
            self._scale_children,
            self._rotate_children,
        ]:
            for i, ob in enumerate(triplet):
                ob.dim = i
        translate_screen.dim = "screen"

        # Attach to the gizmo object
        self.add(*self._line_children)
        self.add(*self._arc_children)
        self.add(translate_screen, *self._translate_children)
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

        # Calculate the direction vectors in ndc space
        directions_in_ndc = [
            center_world.clone()
            .add(self._get_direction(dim))
            .project(camera)
            .sub(center_screen)
            .multiply_scalar(scale_scalar)
            for dim in (0, 1, 2)
        ]
        self._directions_in_screen = [
            (vec.x * canvas_size[0] / 2, vec.y * canvas_size[1] / 2)
            for vec in directions_in_ndc
        ]

        # Determine what directions are orthogonal to the view plane
        # And also a multiplier to compensate for the smaller leverage at a high angle
        show_direction = [True, True, True]
        multipliers = [1, 1, 1]
        for dim, vec in enumerate(directions_in_ndc):
            screen_vec = vec.x * canvas_size[0] / 2, vec.y * canvas_size[1] / 2
            size = (screen_vec[0] ** 2 + screen_vec[1] ** 2) ** 0.5
            show_direction[dim] = size > 15  # in pixels
            multipliers[dim] = 1 / (size + 0.1)

        # Store multipler per direction
        ref_multiplier = min(multipliers)
        self._direction_multipliers = [x / ref_multiplier for x in multipliers]

        # Hide object for which the direction (on screen) becomes ill-defined.
        for dim in (0, 1, 2):
            self._line_children[dim].visible = show_direction[dim]
            self._translate_children[dim].visible = show_direction[dim]
            self._scale_children[dim].visible = show_direction[dim]

        # Determine any flips so that the gizmo faces the camera
        for dim, vec in enumerate(directions_in_ndc):
            if vec.z > 0:
                scale[dim] = -scale[dim]
                dir_screen = self._directions_in_screen[dim]
                self._directions_in_screen[dim] = -dir_screen[0], -dir_screen[1]

        self.scale = Vector3(*scale)

    # %% Utils

    def _get_direction(self, dim):
        """Get a vector indicating the translation direction in world space."""
        assert 0 <= dim <= 2
        if dim == 0:
            return Vector3(1, 0, 0)
        elif dim == 1:
            return Vector3(0, 1, 0)
        else:
            return Vector3(0, 0, 1)

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
            if ob in self._translate_children:
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
        if isinstance(ob.dim, int):
            multiply = self._direction_multipliers[ob.dim]
            direction = self._get_direction(ob.dim).multiply_scalar(multiply)
        else:
            direction = None
        self._ref = {
            "kind": "translate",
            "event": event,
            "pos": self._object_to_control.position.clone(),
            "screen_vecs": self._screen_vecs,
            "dim": ob.dim,
            "direction": direction,
        }

    def _handle_translate_move(self, event):
        dim = self._ref["dim"]
        vec = self._get_world_vector_from_pointer_move(event)
        if dim == "screen":
            position = self._ref["pos"].clone().add_scaled_vector(vec, -1)
        else:
            vec.multiply(self._ref["direction"])
            position = self._ref["pos"].clone().add_scaled_vector(vec, -1)
        self._object_to_control.position = position
        self.position = position.clone()

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
        dir_x, dir_y = self._directions_in_screen[dim]
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
            "screen_vecs": self._screen_vecs,
            "dim": ob.dim,
        }

    def _handle_rotate_move(self, event):
        dim = self._ref["dim"]
        # Calculate axis of rotation
        axis = [0, 0, 0]
        axis[dim] = 1
        # Calculate how far the mouse has moved
        dx = event["x"] - self._ref["event"]["x"]
        dy = event["y"] - self._ref["event"]["y"]
        dir_x, dir_y = self._directions_in_screen[dim]
        dir_norm = (dir_x**2 + dir_y**2) ** 0.5
        dist_pixels = dir_x * dy / dir_norm + dir_y * dx / dir_norm
        # Calculate angle
        angle = dist_pixels / 50
        # Apply
        rot = gfx.linalg.Quaternion().set_from_axis_angle(Vector3(*axis), angle)
        self._object_to_control.rotation = rot.multiply(self._ref["rot"])
