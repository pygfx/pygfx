import numpy as np

from ._base import id_provider
from . import WorldObject, Mesh, Line
from ..resources import Buffer

DOCSTRING_TEMPLATE = """Display a {name} multiple times using instances.

    An instanced {name} with a matrix for each instance.

    Parameters
    ----------
    geometry : Geometry
        The {name}'s geometry data.
    material : Material
        The material with which to render the {name}.
    count : int
        The number of instances to create.
    kwargs : Any
        Additional kwargs get forwarded to the :class:`base class
        <pygfx.objects.{base_cls}>`.

    """


class InstancedObject(WorldObject):
    __doc__ = DOCSTRING_TEMPLATE.format(name="object", base_cls="WorldObject").replace(
        "a object", "an object"
    )

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
        self._store.instance_buffer = Buffer(
            instance_infos, nitems=count, force_contiguous=True
        )
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
        instance_infos = self._store["instance_buffer"].data
        for i in range(len(instance_infos)):
            id_provider.release_id(self, instance_infos[i]["id"])

    @property
    def instance_buffer(self):
        return self._store.instance_buffer

    def set_matrix_at(self, index: int, matrix):
        """set the matrix for the instance at the given index."""
        matrix = np.array(matrix).reshape(4, 4)
        self._store["instance_buffer"].data["matrix"][index] = matrix.T
        self._store["instance_buffer"].update_range(index, 1)

    def get_matrix_at(self, index: int):
        """get the matrix for the instance at the given index."""
        return self._store["instance_buffer"].data["matrix"][index].T

    def _wgpu_get_pick_info(self, pick_value) -> dict:
        info = super()._wgpu_get_pick_info(pick_value)
        # The id maps to one of our instances
        id = pick_value & 1048575  # 2**20-1
        info["instance_index"] = self._idmap.get(id)
        return info


class InstancedMesh(Mesh, InstancedObject):
    __doc__ = DOCSTRING_TEMPLATE.format(name="mesh", base_cls="Mesh")

    def __init__(self, geometry, material, count, **kwargs):
        super().__init__(geometry, material, count, **kwargs)


class InstancedLine(Line, InstancedObject):
    __doc__ = DOCSTRING_TEMPLATE.format(name="line", base_cls="Line")

    def __init__(self, geometry, material, count, **kwargs):
        super().__init__(geometry, material, count, **kwargs)
