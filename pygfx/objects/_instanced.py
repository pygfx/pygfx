import numpy as np

from ._base import id_provider
from . import Mesh
from ..resources import Buffer


class InstancedMesh(Mesh):
    """An instanced mesh with a matrix for each instance."""

    def __init__(self, geometry, material, count, **kwargs):
        super().__init__(geometry, material, **kwargs)
        count = int(count)
        # Create array of `count` instance_info objects
        dtype = np.dtype(
            [
                ("matrix", np.float32, (4, 4)),
                ("id", np.uint32),
                ("_12_bytes_padding", np.uint8, (12,)),
            ]
        )
        instance_infos = np.zeros(count, dtype)
        self.instance_infos = Buffer(instance_infos, nitems=count)
        # Set ids
        self._idmap = {}
        for instance_index in range(count):
            id = id_provider.claim_id(self)
            self._idmap[id] = instance_index
            instance_infos[instance_index]["id"] = id
        # Init eye matrices
        for i in range(4):
            instance_infos["matrix"][:, i, i] = 1

    def __del__(self):
        super().__del__()
        instance_infos = self.instance_infos.data
        for i in range(len(instance_infos)):
            id_provider.release_id(self, instance_infos[i]["id"])

    def set_matrix_at(self, index: int, matrix):
        """set the matrix for the instance at the given index."""
        matrix = np.array(matrix).reshape(4, 4)
        self.instance_infos.data["matrix"][index] = matrix

    def _wgpu_get_pick_info(self, pick_value):
        info = self.material._wgpu_get_pick_info(pick_value)
        # The id maps to one of our instances
        id = pick_value & 1048575  # 2**20-1
        info["instance_index"] = self._idmap.get(id)
        return info
