import numpy as np

from collections import namedtuple


BufferData = namedtuple("BufferData", ["array", "mapped"], defaults=[False])


class Geometry:
    """ A geometry represents the (input) data of mesh, line, or point
    geometry. It includes vertex positions, face indices, normals,
    colors, UVs, and custom attributes within buffers. It can also be thought
    of as a datasource.

    Subclasses can implement a convenient way to generate data for
    specific shapes, but can also provide advanced techniques to manage
    and generate data.

    For the GPU, geometry objects are responsible for providing buffers.
    """

    def __init__(self, index=None, *vertex_data):
        # todo: don't cast if an array is already given
        self.index = None if index is None else np.array(index, np.uint32)

        self.vertex_data = []
        self.bindings = {
            slot: (np.array(array, np.float32), False) for array in vertex_data
        }
        # self.storage_data = {} ?
