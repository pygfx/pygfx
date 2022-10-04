import math

from pygfx.linalg.vector3 import Vector3

from .. import (
    sphere_geometry,
    Mesh,
    MeshBasicMaterial,
    Geometry,
    LineArrowMaterial,
    Line,
    WorldObject,
    LineThinSegmentMaterial,
    LineThinMaterial,
)


class PointLightHelper(Mesh):
    """Helper class to visualize a point light using a sphere (by default)."""

    def __init__(self, light, size=1, geometry=None, color=None):
        if geometry is None:
            geometry = sphere_geometry(size)

        material = MeshBasicMaterial()

        super().__init__(geometry, material)

        self.light = light
        self.light.update_matrix_world()

        self.color = color

        self._matrix = self.light.matrix_world
        self.matrix_auto_update = False

        self.update()

    def update(self):
        if self.color:
            self.material.color = self.color
        else:
            self.material.color = self.light.color

    def update_matrix_world(
        self, force=False, update_children=True, update_parents=False
    ):
        self.update()
        self._matrix_world_dirty = True
        super().update_matrix_world(force, update_children, update_parents)


class DirectionalLightHelper(WorldObject):
    def __init__(self, light, length=None, color=None):
        super().__init__()

        self.light = light
        self.light.update_matrix_world()

        self.color = color
        self._matrix = self.light.matrix_world
        self.matrix_auto_update = False

        self.lines = Line(
            Geometry(
                positions=[
                    [1, 0, 0],
                    [1, 0, 1],
                    [-1, 0, 0],
                    [-1, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [0, -1, 0],
                    [0, -1, 1],
                ]
            ),
            LineArrowMaterial(color=light.color.hex),
        )
        length = length or 1
        self.lines.scale.set(length / 5, length / 5, length)
        self.add(self.lines)
        self.update()

    def update(self):
        if self.color:
            self.lines.material.color = self.color
        else:
            self.lines.material.color = self.light.color

        _tmp_vector.set_from_matrix_position(self.light.target.matrix_world)

        _update_matrix_world = self.update_matrix_world
        self.update_matrix_world = lambda *args, **kwargs: None
        self.lines.look_at(_tmp_vector)
        self.update_matrix_world = _update_matrix_world

    def update_matrix_world(
        self, force=False, update_children=True, update_parents=False
    ):
        self.update()
        self._matrix_world_dirty = True
        super().update_matrix_world(force, update_children, update_parents)


class DirectionalLightShadowHelper(WorldObject):
    def __init__(self, light, size=None, color=None):
        super().__init__()

        self.light = light
        self.light.update_matrix_world()

        self.color = color

        self._matrix = self.light.matrix_world
        self.matrix_auto_update = False

        if size is None:
            half_w = light.shadow.camera.width / 2
            half_h = light.shadow.camera.height / 2
        else:
            half_w = size / 2
            half_h = size / 2

        geometry = Geometry(
            positions=[
                [-half_w, half_h, 0],
                [half_w, half_h, 0],
                [half_w, -half_h, 0],
                [-half_w, -half_h, 0],
                [-half_w, half_h, 0],
            ]
        )
        self._material = LineThinMaterial()

        self.light_plane = Line(geometry, self._material)
        self.add(self.light_plane)

        self.target_line = Line(
            Geometry(positions=[[0, 0, 0], [0, 0, 1]]), self._material
        )
        self.add(self.target_line)

        self.update()

    def update(self):
        if self.color:
            self._material.color = self.color
        else:
            self._material.color = self.light.color

        _tmp_vector.set_from_matrix_position(self.light.target.matrix_world)
        _tmp_vector2.set_from_matrix_position(self.light.matrix_world)
        _tmp_vector3.sub_vectors(_tmp_vector, _tmp_vector2)

        _update_matrix_world = self.update_matrix_world
        self.update_matrix_world = lambda *args, **kwargs: None
        self.light_plane.look_at(_tmp_vector)
        self.target_line.look_at(_tmp_vector)
        self.target_line.scale.z = _tmp_vector3.length()
        self.update_matrix_world = _update_matrix_world

    def update_matrix_world(
        self, force=False, update_children=True, update_parents=False
    ):
        self.update()
        self._matrix_world_dirty = True
        super().update_matrix_world(force, update_children, update_parents)


class SpotLightHelper(WorldObject):
    def __init__(self, light, color=None):
        super().__init__()

        self.light = light
        self.light.update_matrix_world()

        self.color = color

        self._matrix = self.light.matrix_world
        self.matrix_auto_update = False

        positions = [
            [0, 0, 0],
            [0, 0, -1],
            [0, 0, 0],
            [1, 0, -1],
            [0, 0, 0],
            [-1, 0, -1],
            [0, 0, 0],
            [0, 1, -1],
            [0, 0, 0],
            [0, -1, -1],
        ]

        for i in range(32):
            p1 = i / 32 * math.pi * 2
            p2 = (i + 1) / 32 * math.pi * 2

            positions.append([math.cos(p1), math.sin(p1), -1])
            positions.append([math.cos(p2), math.sin(p2), -1])

        geometry = Geometry(positions=positions)

        material = LineThinSegmentMaterial(thickness=1.0)

        self.cone = Line(geometry, material)

        self.add(self.cone)
        self.update()

    def update(self):
        cone_length = self.light.distance if self.light.distance else 1000
        cone_width = cone_length * math.tan(self.light.angle)

        self.cone.scale.set(cone_width, cone_width, cone_length)

        _tmp_vector.set_from_matrix_position(self.light.target.matrix_world)

        _update_matrix_world = self.update_matrix_world
        self.update_matrix_world = lambda *args, **kwargs: None
        self.cone.look_at(_tmp_vector)
        self.update_matrix_world = _update_matrix_world

        if self.color:
            self.cone.material.color = self.color
        else:
            self.cone.material.color = self.light.color

    def update_matrix_world(
        self, force=False, update_children=True, update_parents=False
    ):
        self.update()
        self._matrix_world_dirty = True
        super().update_matrix_world(force, update_children, update_parents)


_tmp_vector = Vector3()
_tmp_vector2 = Vector3()
_tmp_vector3 = Vector3()
