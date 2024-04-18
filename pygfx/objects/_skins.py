import numpy as np
import pylinalg as la
from typing import List
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

    class __Transform:
        position: np.ndarray
        rotation: np.ndarray
        scale: np.ndarray
        matrix: np.ndarray

    def __init__(self, name=""):
        super().__init__(name=name)

        self._children: "List[Bone]" = []
        self._parent: "Bone" = None

        self.visible = False

        self.__matrix_world_needs_update = False

        # compatible with cuurrent WorldObject RecursiveTransform system, but does nothing.
        # temporary solution before we refactor the WorldObject transform system.
        # See: https://github.com/pygfx/pygfx/pull/715#issuecomment-2053385803
        self.world = Bone.__Transform()
        self.world.matrix = np.eye(4)
        self.local = Bone.__Transform()
        self.local.position = np.zeros(3)
        self.local.rotation = np.zeros(4)
        self.local.scale = np.ones(3)
        self.local.matrix = np.eye(4)

    @property
    def parent(self):
        if self._parent is None:
            return None
        if isinstance(self._parent, Bone):
            return self._parent
        else:
            return self._parent()

    def update_matrix(self):
        self.local.matrix = la.mat_compose(
            self.local.position, self.local.rotation, self.local.scale
        )
        self.__matrix_world_needs_update = True

    def update_matrix_world(self):
        self.update_matrix()

        if self.__matrix_world_needs_update:
            if self.parent is not None:
                self.world.matrix = self.parent.world.matrix @ self.local.matrix
            else:
                self.world.matrix = self.local.matrix
            self.__matrix_world_needs_update = False

        for child in self._children:
            child.update_matrix_world()

    def add(self, *bones: "Bone") -> "Bone":
        for obj in bones:
            if obj == self:
                # can't add self as a child
                continue

            if obj and isinstance(obj, Bone):
                if obj._parent is not None:
                    obj._parent.remove(obj)

                obj._parent = self
                self._children.append(obj)
            else:
                pass
                # Now the Bone class has specific logic, so we just pass if it's not a Bone

        return self

    def __repr__(self) -> str:
        return f"Bone {self.name} {self.local.position} {self.local.rotation}\n"


class Skeleton:
    """A skeleton.

    A skeleton is a collection of bones that are used to animate a SkinnedMesh.

    """

    def __init__(self, bones: List[Bone], bone_inverses=None):
        super().__init__()
        if bones is None:
            bones = []
        self.bones = bones[:]
        if bone_inverses is None:
            bone_inverses = []
        self.bone_inverses = bone_inverses

        count = len(self.bones)
        self.bone_matrices_buffer = Buffer(
            array_from_shadertype(
                {
                    "bone_matrices": "4x4xf4",
                },
                count,
            )
        )

        if len(self.bone_inverses) == 0:
            self.calculate_inverses()

    def calculate_inverses(self):
        """Generate the bone_inverses array if not provided in the constructor."""
        self.bone_inverses.clear()
        for bone in self.bones:
            bone.update_matrix_world()
            self.bone_inverses.append(np.linalg.inv(bone.world.matrix))

    def pose(self):
        """Reset the skeleton to the binding-time pose."""
        for i, bone in enumerate(self.bones):
            bone.world.matrix = np.linalg.inv(self.bone_inverses[i])

        for bone in self.bones:
            if bone.parent and isinstance(bone.parent, Bone):
                bone.local.matrix = (
                    np.linalg.inv(bone.parent.world.matrix) @ bone.world.matrix
                )
            else:
                bone.local.matrix = bone.world.matrix

            bone.local.position, bone.local.rotation, bone.local.scale = (
                la.mat_decompose(bone.local.matrix)
            )

    def update(self):
        # TODO: we should update the bone matrices buffer automatically by some mechanism.
        # See: https://github.com/pygfx/pygfx/pull/715#issuecomment-2046493145
        """Update the bone matrices buffer."""

        # TODO: update bone matrices from root to leaf，it's a temporary solution.
        # See: https://github.com/pygfx/pygfx/pull/715#issuecomment-2053385803
        for bone in self.bones:
            if bone.parent and isinstance(bone.parent, Bone):
                continue
            bone.update_matrix_world()

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
        """The base matrix that is used for the bound bone transforms."""
        return self._bind_matrix

    @bind_matrix.setter
    def bind_matrix(self, value):
        self._bind_matrix = value
        self.uniform_buffer.data["bind_matrix"] = self._bind_matrix.T
        self.uniform_buffer.update_range()

    @property
    def bind_matrix_inv(self):
        """The base matrix that is used for resetting the bound bone transforms."""
        return self._bind_matrix_inv

    @bind_matrix_inv.setter
    def bind_matrix_inv(self, value):
        self._bind_matrix_inv = value
        self.uniform_buffer.data["bind_matrix_inv"] = self._bind_matrix_inv.T
        self.uniform_buffer.update_range()

    @property
    def bind_mode(self):
        """
        Either "attached" or "detached".
        "attached" means the skinned mesh shares the same world space as the skeleton.
        In contrast, "detached" is useful when sharing a skeleton across multiple skinned meshes.
        Default is "attached".
        """
        return self._bind_mode

    @bind_mode.setter
    def bind_mode(self, value):
        assert value in ["attached", "detached"]
        self._bind_mode = value

    @callback
    def _update_uniform_buffers(self, transform: AffineBase):
        super()._update_uniform_buffers(transform)

        if self.bind_mode == "attached":
            self.bind_matrix_inv = self.world.inverse_matrix
        elif self.bind_mode == "detached":
            self.bind_matrix_inv = np.linalg.inv(self.bind_matrix)

    def bind(self, skeleton: Skeleton, bind_matrix=None):
        """Bind a skeleton to the skinned mesh.
        The bind_matrix gets saved to .bind_matrix property and the .bind_matrix_inv gets calculated.

        Args:
            skeleton (Skeleton): The skeleton created from the bones tree.
            bind_matrix (np.ndarray, optional): The base transform of the skeleton.
        """
        self.skeleton = skeleton

        if bind_matrix is None:
            self.skeleton.calculate_inverses()
            bind_matrix = self.world.matrix

        self.bind_matrix = bind_matrix
        self.bind_matrix_inv = np.linalg.inv(bind_matrix)

    def pose(self):
        """Reset the skinned mesh to the binding-time pose."""
        self.skeleton.pose()
