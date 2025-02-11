import pylinalg as la
from typing import List
from .. import Geometry, Line, LineSegmentMaterial, Bone, WorldObject


class SkeletonHelper(Line):
    """A helper object to assist with visualizing a Skeleton"""

    def __init__(self, wobject: WorldObject, thickness=1.0):
        bones = self._get_bones(wobject)
        positions = []
        colors = []
        if not bones:
            positions.append([0, 0, 0])
            colors.append([0, 0, 0])
        else:
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
        self.bone_descendants = [
            b for b in bones if b.parent and isinstance(b.parent, Bone)
        ]

        self.local.matrix = wobject.world.matrix

    def _update_object(self):
        # Update on every draw: the helper matrix always follows the root object
        self.local.matrix = self.root.world.matrix
        super()._update_object()
        self.update()

    def update(self):
        # TODO: we should update it automatically by some mechanism.
        # See: https://github.com/pygfx/pygfx/pull/715#issuecomment-2046493145
        """Update the helper object to match the Skeleton's bones."""
        geometry = self.geometry
        root_matrix_world_inv = self.root.world.inverse_matrix
        positions = geometry.positions
        positions_data = positions.data

        j = 0
        for bone in self.bone_descendants:
            bone_parent = bone.parent
            bone_matrix = root_matrix_world_inv @ bone.world.matrix
            la.mat_decompose_translation(bone_matrix, out=positions_data[j])

            parent_bone_matrix = root_matrix_world_inv @ bone_parent.world.matrix
            la.mat_decompose_translation(parent_bone_matrix, out=positions_data[j + 1])

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
