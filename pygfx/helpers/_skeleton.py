import pylinalg as la
from typing import List
from .. import Geometry, Line, LineSegmentMaterial, Bone, WorldObject


class SkeletonHelper(Line):
    """A helper object to assist with visualizing a Skeleton"""

    def __init__(self, wobject: WorldObject, thickness=1.0):

        bones = self._get_bones(wobject)

        positions = []
        colors = []
        for bone in bones:
            if bone.parent and isinstance(bone.parent, Bone):
                positions.append([0, 0, 0])
                positions.append([0, 0, 0])

                colors.append([0, 0, 1])
                colors.append([0, 1, 0])

        super().__init__(
            Geometry(positions=positions, colors=colors),
            LineSegmentMaterial(
                thickness=thickness, color_mode="vertex", depth_test=False
            ),
        )

        self.root = wobject
        self.bones = bones

        self.local.matrix = wobject.world.matrix

        # the helper matrix always follows the root object
        def _update_matrix(*args):
            self.local.matrix = self.root.world.matrix

        self.root.world.on_update(_update_matrix)

    def update(self):
        # TODO: we should update it automatically by some mechanism.
        # See: https://github.com/pygfx/pygfx/pull/715#issuecomment-2046493145
        """Update the helper object to match the Skeleton's bones."""
        bones = self.bones
        geometry = self.geometry
        root_matrix_world_inv = self.root.world.inverse_matrix
        positions = geometry.positions

        j = 0

        for bone in bones:
            if bone.parent and isinstance(bone.parent, Bone):
                bone_matrix = root_matrix_world_inv @ bone.world.matrix
                positions.data[j] = la.mat_decompose_translation(bone_matrix)

                parent_bone_matrix = root_matrix_world_inv @ bone.parent.world.matrix
                positions.data[j + 1] = la.mat_decompose_translation(parent_bone_matrix)

                j += 2

        positions.update_range()

    def _get_bones(self, obj) -> List[Bone]:
        # Recursively get all bones from the object and its children.
        bones = []
        if isinstance(obj, Bone):
            bones.append(obj)

        for child in obj.children:
            bones.extend(self._get_bones(child))

        return bones
