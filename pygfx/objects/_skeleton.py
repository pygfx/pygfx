import numpy as np
from ._base import WorldObject
from ..utils import array_from_shadertype
from ..utils.transform import AffineBase, callback
from ..resources import Buffer
from ._more import Mesh


class Bone(WorldObject):
    """A bone.

    A bone is a part of a skeleton.The skeleton in turn is used by the SkinnedMesh.
    Bones are almost identical to a blank WorldObject.

    """

    def __init__(self, name=""):
        super().__init__()
        self.name = name

    def __repr__(self) -> str:
        return f"Bone {self.name} {self.local.position} {self.local.rotation}\n"


class Skeleton:
    """A skeleton.

    A skeleton is a collection of bones that are used to animate a SkinnedMesh.

    """

    uniform_type = dict(
        bone_matrices="4x4xf4",
    )

    def __init__(self, bones: list[Bone] = None, bone_inverses=None):
        super().__init__()
        if bones is None:
            bones = []
        self.bones = bones[:]
        self.bone_inverses = bone_inverses or []

        self.init()

    def init(self):
        count = len(self.bones)
        self.bone_matrices_buffer = Buffer(
            array_from_shadertype(self.uniform_type, count)
        )

        if len(self.bone_inverses) == 0:
            self.calculate_inverses()

    def calculate_inverses(self):
        self.bone_inverses.clear()
        for bone in self.bones:
            self.bone_inverses.append(bone.world.inverse_matrix)

    def update(self):
        for i, bone in enumerate(self.bones):
            self.bone_matrices_buffer.data[i]["bone_matrices"] = (
                bone.world.matrix @ self.bone_inverses[i]
            ).T

        self.bone_matrices_buffer.update_range()

    def get_bone(self, name):
        for bone in self.bones:
            if bone.name == name:
                return bone
        return None


class SkinnedMesh(Mesh):
    """A skinned mesh.

    A mesh that has a Skeleton with bones that can then be used to animate the vertices of the geometry.

    """

    uniform_type = dict(
        Mesh.uniform_type,
        bind_matrix="4x4xf4",
        bind_matrix_inv="4x4xf4",
    )

    def __init__(self, geometry, material):
        super().__init__(geometry, material)

        self.bind_matrix = np.eye(4)
        self.bind_matrix_inv = np.eye(4)

        self._bind_mode = "attached"

    @property
    def bind_matrix(self):
        return self._bind_matrix

    @bind_matrix.setter
    def bind_matrix(self, value):
        self._bind_matrix = value
        self.uniform_buffer.data["bind_matrix"] = self._bind_matrix.T
        self.uniform_buffer.update_range()

    @property
    def bind_matrix_inv(self):
        return self._bind_matrix_inv

    @bind_matrix_inv.setter
    def bind_matrix_inv(self, value):
        self._bind_matrix_inv = value
        self.uniform_buffer.data["bind_matrix_inv"] = self._bind_matrix_inv.T
        self.uniform_buffer.update_range()

    @property
    def bind_mode(self):
        return self._bind_mode

    @bind_mode.setter
    def bind_mode(self, value):
        self._bind_mode = value

    @callback
    def _update_uniform_buffers(self, transform: AffineBase):
        super()._update_uniform_buffers(transform)

        if self.bind_mode == "attached":
            self.bind_matrix_inv = self.world.inverse_matrix
        elif self.bind_mode == "detached":
            self.bind_matrix_inv = np.linalg.inv(self.bind_matrix)

    def bind(self, skeleton: Skeleton, bind_matrix=None):
        self.skeleton = skeleton

        if bind_matrix is None:
            self.skeleton.calculate_inverses()
            bind_matrix = self.world.matrix

        self.bind_matrix = bind_matrix
        self.bind_matrix_inv = np.linalg.inv(bind_matrix)
