import numpy as np

from ..utils.trackable import Trackable
from ..resources import Resource, Buffer, Texture
from ..linalg.utils import aabb_to_sphere


class Geometry(Trackable):
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
    * ``colors``: Per-vertex colors. Must be NxM, with M 1-4 for gray,
      gray+alpha, rgb, rgba, respectively.
    * ``sizes``: Scalar size per-vertex.
    * ``grid``: A 2D or 3D Texture/TextureView that contains a regular grid of
      data. I.e. for images and volumes.

    :Example:

    .. code-block:: py

        g = Geometry(positions=[[1, 2], [2, 4], [3, 5], [4, 1]])
        g.positions.data  # numpy array

    """

    def __init__(self, **kwargs):
        super().__init__()

        self._aabb = None
        self._aabb_rev = None
        self._bsphere = None
        self._bsphere_rev = None

        for name, val in kwargs.items():

            # Get resource object
            if isinstance(val, Resource):
                resource = val
            else:
                if not isinstance(val, np.ndarray):
                    dtype = np.uint32 if name == "indices" else np.float32
                    val = np.asanyarray(val, dtype=dtype)
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

    def __setattr__(self, key, new_value):
        if not key.startswith("_"):
            if isinstance(new_value, Trackable) or key in self._store:
                return setattr(self._store, key, new_value)
        object.__setattr__(self, key, new_value)

    def __getattribute__(self, key):
        if not key.startswith("_"):
            if key in self._store:
                return getattr(self._store, key)
        return object.__getattribute__(self, key)

    def __dir__(self):
        x = object.__dir__(self)
        x.extend(dict.keys(self._store))
        return x

    def bounding_box(self):
        """Compute the axis-aligned bounding box based on either positions
        or the shape of the grid buffer.

        If both are present, the bounding box will be computed based on
        the positions buffer.
        """
        if hasattr(self, "positions"):
            if self._aabb_rev == self.positions.rev:
                return self._aabb
            pos = self.positions.data
            self._aabb = np.array([pos.min(axis=0), pos.max(axis=0)])
            # If positions contains xy, but not z, assume z=0
            if self._aabb.shape[1] == 2:
                self._aabb = np.column_stack([self._aabb, np.zeros((2, 1), np.float32)])
            self._aabb_rev = self.positions.rev
            return self._aabb

        if hasattr(self, "grid"):
            if self._aabb_rev == self.grid.rev:
                return self._aabb
            # account for multi-channel image data
            if hasattr(self.grid, "texture"):
                # self.grid can be a TextureView instead of a Texture, so in that
                # case, get the shape from the texture attribute
                grid_shape = self.grid.texture.data.shape[: self.grid.texture.dim]
            else:
                grid_shape = self.grid.data.shape[: self.grid.dim]
            # create aabb in index/data space
            aabb = np.array([np.zeros_like(grid_shape), grid_shape], dtype="f8")
            # convert to local image space by aligning
            # center of voxel index (0, 0, 0) with origin (0, 0, 0)
            aabb -= 0.5
            # ensure coordinates are 3D
            # NOTE: important we do this last, we don't want to apply
            # the -0.5 offset to the z-coordinate of 2D images
            if aabb.shape[1] == 2:
                aabb = np.hstack([aabb, [[0], [0]]])
            self._aabb = aabb
            self._aabb_rev = self.grid.rev
            return self._aabb

        raise ValueError(
            "No positions or grid buffer available for bounding volume computation"
        )

    def bounding_sphere(self):
        """Compute the bounding sphere based on the axis-aligned bounding box.

        Note: not the optimal fit.
        """
        if self._bsphere is not None and self._bsphere_rev == self._aabb_rev:
            return self._bsphere

        self._bsphere = aabb_to_sphere(self.bounding_box())
        self._bsphere_rev = self._aabb_rev
        return self._bsphere
