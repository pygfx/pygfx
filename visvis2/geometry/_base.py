import numpy as np

from collections import namedtuple


BufferData = namedtuple("BufferData", ["array", "mapped"], defaults=[False])


class Geometry:
    def __init__(self, index=None, *vertex_data):
        # todo: don't cast if an array is already given
        self.index = None if index is None else np.array(index, np.uint32)
        self.vertex_data = [np.array(array, np.float32) for array in vertex_data]
        # self.storage_data = {} ?

