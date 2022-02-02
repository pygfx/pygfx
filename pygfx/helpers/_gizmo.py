"""
A transform gizmo to manipulate world objects.
"""


import numpy as np
import pygfx as gfx
from ..objects import WorldObject
from pygfx.controls._orbit import get_screen_vectors_in_world_cords


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

        self._screen_size = float(screen_size)
        self._renderer = None
        self._canvas = None
        self._camera = None
        self._ref = None
        self._vec1 = gfx.linalg.Vector3(0, 0, 0)
        self._vec2 = gfx.linalg.Vector3(0, 0, 0)

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

        # Create lines
        line_geo = gfx.Geometry(positions=[[0, 0, 0], [1, 0, 0]])
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
        self._line_children = line_x, line_y, line_z

        # Create translate handles
        cone_geo = gfx.cone_geometry(0.1, 0.17)
        cone_geo.positions.data[:] = cone_geo.positions.data[:, ::-1]
        trans_x = gfx.Mesh(
            cone_geo,
            gfx.MeshBasicMaterial(color="#ff0000"),
        )
        trans_y = gfx.Mesh(
            cone_geo,
            gfx.MeshBasicMaterial(color="#00ff00"),
        )
        trans_z = gfx.Mesh(
            cone_geo,
            gfx.MeshBasicMaterial(color="#0000ff"),
        )
        self._trans_children = trans_x, trans_y, trans_z

        # Position the translate handles
        # todo: I don't understand the need for the minus in z :/
        trans_x.position.set(1, 0, 0)
        trans_y.position.set(0, 1, 0)
        trans_z.position.set(0, 0, -1)

        # Store info on the object that we can use in the event handler
        trans_x.direction = gfx.linalg.Vector3(1, 0, 0)
        trans_y.direction = gfx.linalg.Vector3(0, 1, 0)
        trans_z.direction = gfx.linalg.Vector3(0, 0, 1)

        # Rotate objects to their correct orientation
        for ob in [line_y, trans_y]:
            ob.rotation.set_from_axis_angle(gfx.linalg.Vector3(0, 0, 1), np.pi / 2)
        for ob in [line_z, trans_z]:
            ob.rotation.set_from_axis_angle(gfx.linalg.Vector3(0, 1, 0), np.pi / 2)

        # Attach to the gizmo object
        self.add(line_x, line_y, line_z)
        self.add(trans_x, trans_y, trans_z)

    def update_matrix_world(self, *args, **kwargs):
        # This gets called by the renderer just before rendering.
        # We take this moment to scale the gizmo.
        if self._object_to_control and self._canvas and self._camera:
            # Get the relation between screen space and world space, and store it
            vec1, vec2 = get_screen_vectors_in_world_cords(
                self._object_to_control.position,
                self._canvas.get_logical_size(),
                self._camera,
            )
            self._vec1, self._vec2 = vec1, vec2
            # Scale this object
            scale = self._screen_size * (vec1.length() + vec2.length()) / 2
            self.scale = gfx.linalg.Vector3(scale, scale, scale)
        # Update the matrix (including our scale change)
        super().update_matrix_world(*args, **kwargs)

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
            print(ob)
            if ob in self._trans_children:
                self._handle_translate_start(event, ob)
        elif type == "pointer_up":
            self._ref = None
        elif type == "pointer_move" and self._ref:
            if self._ref["kind"] == "translate":
                self._handle_translate_move(event)

    def _handle_translate_start(self, event, ob):
        self._ref = {
            "kind": "translate",
            "event": event,
            "pos": self._object_to_control.position.clone(),
            "direction": ob.direction,
        }

    def _get_world_vector_from_pointer_move(self, event):
        dx = event["x"] - self._ref["event"]["x"]
        dy = event["y"] - self._ref["event"]["y"]
        return (
            self._vec1.clone().multiply_scalar(-dx).add_scaled_vector(self._vec2, +dy)
        )

    def _handle_translate_move(self, event):
        vec = self._get_world_vector_from_pointer_move(event)
        vec.multiply(self._ref["direction"])
        position = self._ref["pos"].clone().add_scaled_vector(vec, -1)
        self._object_to_control.position = position
        self.position = position
