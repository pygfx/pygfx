import numpy as np

from ..objects._base import ResourceContainer
from ..resources import Buffer


class Geometry(ResourceContainer):
    """A geometry represents the (input) data of mesh, line, or point
    geometry. It can include vertex positions, normals, colors, uvs,
    and custom data buffers. Face indices can be given using `index`.

    Buffer data can be provided as kwargs, these are converted to numpy arrays
    (if necessary) and wrapped in a Buffer.

    Example:

        g = Geometry(positions=[[1, 2], [2, 4], [3, 5], [4, 1]])
        g.positions.data  # numpy array

    """

    def __init__(self, **data):
        super().__init__()
        for name, val in data.items():
            if not isinstance(val, np.ndarray):
                val = np.asanyarray(val, dtype=np.float32)
            elif val.dtype == np.float64:
                raise ValueError(
                    "64-bit float is not supported, use 32-bit floats instead"
                )
            if name == "index":
                usage = "index|storage"
                # No assumptions about shape; they're considered flat anyway
            elif name == "positions":
                usage = "vertex|storage"
                if not (val.ndim == 2 and val.shape[-1] == 3):
                    raise ValueError("Expected Nx3 data for positions")
            elif name == "normals":
                usage = "vertex|storage"
                if not (val.ndim == 2 and val.shape[-1] == 3):
                    raise ValueError("Expected Nx3 data for normals")
            elif name == "texcoords":
                usage = "vertex|storage"
                if val.ndim == 1:
                    pass
                elif not (val.ndim == 2 and val.shape[-1] in (1, 2, 3)):
                    raise ValueError("Expected Nx1, Nx2 or Nx3 data for texcoords")
            else:
                usage = "vertex|storage"
            setattr(self, name, Buffer(val, usage=usage))
