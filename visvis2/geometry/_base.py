import numpy as np

from .._wrappers import BufferWrapper


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

    def __init__(self, **data):
        for name, val in data.items():
            val = np.asanyarray(val)
            usage = None
            if name.lower() in ("index", "indices"):
                usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.INDEX
            setattr(name, BufferWrapper(val, usage=usage))
