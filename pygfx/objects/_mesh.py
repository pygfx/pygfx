import numpy as np

from ._base import WorldObject
from ..datawrappers import Buffer


class Mesh(WorldObject):
    def __init__(self, geometry, material):
        super().__init__()
        self.geometry = geometry
        self.material = material


class InstancedMesh(Mesh):
    def __init__(self, geometry, material, count):
        super().__init__(geometry, material)
        count = int(count)
        # Create count eye matrices
        matrices = np.zeros((count, 4, 4), np.float32)
        for i in range(4):
            matrices[:, i, i] = 1
        self.matrices = Buffer(matrices, usage="STORAGE", nitems=count)

    def set_matrix_at(self, index: int, matrix):
        matrix = np.array(matrix).reshape(4, 4)
        self.matrices.data[index] = matrix
