import numpy as np

from ..objects._base import ResourceContainer
from ..resources import Resource, Buffer, Texture


class Geometry(ResourceContainer):
    """A Geomerty object is a container for geometry data of a WorldObject.

    A geometry object contains the data that defines (the shape of) the
    object, such as positions, plus data associated with these positions
    (normals, texcoords, colors, etc.).

    Its attributes are Buffer and Texture(View) objects. The data can
    be provided as keyword arguments, which are converted to numpy
    arrays and wrapped in a Buffer if necessary.

    The names for these attributes are standardized so that the
    renderers know what to expect. Each material requires certain
    attributes to be present, and may support optional attributes.
    Optional attributes must always be "turned on" on the material;
    their presence on the geometry does not mean that they're used.

    The standardized names are:

    * ``indices``: An index into per-vertex data. Typically Nx3 for mesh geometry.
    * ``positions``: Nx3 positions (xyz), defining the location of e.g. vertices or points.
    * ``normals``: Nx3 normal vectors. These may or may not be unit.
    * ``texcoords``: Texture coordinates used to lookup the color for a vertex.
      Can be Nx1, Nx2 or Nx3, corresponding to a 1D, 2D and 3D texture map.
    * ``colors``: Nx4 per-vertex colors (rgba).
    * ``sizes``: Scalar size per-vertex.
    * ``grid``: A 2D or 3D Texture/TextureView that contains a regular grid of
      data. I.e. for images and volumes.

    Example:

        g = Geometry(positions=[[1, 2], [2, 4], [3, 5], [4, 1]])
        g.positions.data  # numpy array

    """

    def __init__(self, **kwargs):
        super().__init__()

        for name, val in kwargs.items():

            # Get resource object
            if isinstance(val, Resource):
                resource = val
            else:
                if not isinstance(val, np.ndarray):
                    val = np.asanyarray(val, dtype=np.float32)
                if val.dtype == np.float64:
                    raise ValueError(
                        "64-bit float is not supported, use 32-bit floats instead"
                    )
                if name == "grid":
                    dim = val.ndim
                    if dim > 2 and val.shape[-1] <= 4:
                        dim -= 1  # last array dim is probably (a subset of) rgba
                    resource = Texture(val, dim=dim).get_view()
                else:
                    resource = Buffer(val)

            # Checks
            if isinstance(resource, Buffer):
                format = resource.format
                if name == "indices":
                    pass  # No assumptions about shape; they're considered flat anyway
                elif name == "positions":
                    if not format.startswith("3x"):
                        raise ValueError("Expected Nx3 data for positions")
                elif name == "normals":
                    if not format.startswith("3x"):
                        raise ValueError("Expected Nx3 data for normals")
                elif name == "texcoords":
                    if not ("x" not in format or format.startswith(("1x", "2x", "3x"))):
                        raise ValueError("Expected Nx1, Nx2 or Nx3 data for texcoords")
                elif name == "colors":
                    if not format.startswith("4x"):
                        raise ValueError("Expected Nx4 data for colors")
                elif name == "sizes":
                    if not ("x" not in format):
                        raise ValueError("Expected array of scalars for sizes")
                else:
                    pass  # Unknown attribute - no checks

            # Store
            setattr(self, name, resource)
