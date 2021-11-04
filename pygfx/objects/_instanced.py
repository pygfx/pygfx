import numpy as np

from . import Mesh
from ..resources import Buffer


class InstancedMesh(Mesh):
    """An instanced mesh with a matrix for each instance."""

    def __init__(self, geometry, material, count):
        super().__init__(geometry, material)
        count = int(count)
        # Create count eye matrices
        matrices = np.zeros((count, 4, 4), np.float32)
        for i in range(4):
            matrices[:, i, i] = 1
        self.matrices = Buffer(matrices, nitems=count)

    def set_matrix_at(self, index: int, matrix):
        """set the matrix for the instance at the given index."""
        matrix = np.array(matrix).reshape(4, 4)
        self.matrices.data[index] = matrix
