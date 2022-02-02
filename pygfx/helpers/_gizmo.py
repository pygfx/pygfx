"""
A transform gizmo to manipulate world objects.
"""


import numpy as np
import pygfx as gfx
from ..objects import WorldObject
from pygfx.controls._orbit import get_screen_vectors_in_world_cords


class TransformGizmo(WorldObject):
    def __init__(self, object_to_control):
        super().__init__()
        self._create_components()
        self._scale()

        self._object_to_control = object_to_control
        self._ref = None

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

    def _scale(self):
        scale = 1

        self._line_children[0].scale.set(scale, 1, 1)
        self._line_children[1].scale.set(1, scale, 1)
        self._line_children[2].scale.set(1, 1, scale)

        self._trans_children[0].position.set(scale, 0, 0)
        self._trans_children[1].position.set(0, scale, 0)
        self._trans_children[2].position.set(0, 0, -scale)
        # todo: I don't understand the need for the minus here :/

    def add_default_event_handlers(self, renderer, camera):
        canvas = renderer.target
        canvas.add_event_handler(
            lambda event: self._handle_event(event, renderer, canvas, camera),
            "pointer_down",
            "pointer_move",
            "pointer_up",
        )

    def _handle_event(self, event, renderer, canvas, camera):
        # todo: check buttons and modifiers
        type = event["event_type"]
        if type in "pointer_down":
            if event["modifiers"]:
                return
            self._ref = None
            # todo: make renderer cache pick info calls for the same frame
            info = renderer.get_pick_info((event["x"], event["y"]))
            ob = info["world_object"]
            print(ob)
            if ob in self._trans_children:
                self._handle_translate_start(event, canvas, camera, ob)
        elif type == "pointer_up":
            self._ref = None
        elif type == "pointer_move" and self._ref:
            if self._ref["kind"] == "translate":
                self._handle_translate_move(event)

    def _handle_translate_start(self, event, canvas, camera, ob):
        vecx, vecy = get_screen_vectors_in_world_cords(
            self._object_to_control.get_world_position(),
            canvas.get_logical_size(),
            camera,
        )
        self._ref = {
            "kind": "translate",
            "event": event,
            "pos": self._object_to_control.position.clone(),
            "vecx": vecx,
            "vecy": vecy,
            "direction": ob.direction,
        }

    def _get_world_vector_from_pointer_move(self, event):
        dx = event["x"] - self._ref["event"]["x"]
        dy = event["y"] - self._ref["event"]["y"]
        return (
            self._ref["vecx"]
            .clone()
            .multiply_scalar(-dx)
            .add_scaled_vector(self._ref["vecy"], +dy)
        )

    def _handle_translate_move(self, event):
        vec = self._get_world_vector_from_pointer_move(event)
        vec.multiply(self._ref["direction"])
        position = self._ref["pos"].clone().add_scaled_vector(vec, -1)
        self._object_to_control.position = position
        self.position = position
