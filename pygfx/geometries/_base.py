import numpy as np

from ..utils.trackable import Trackable
from ..resources import Resource, Buffer, Texture
from ..linalg.utils import aabb_to_sphere


class Geometry(Trackable):
    """Generic container for Geometry data.

    Parameters
    ----------
    kwargs : dict
        A dict of attributes to define on the geometry object. Keys must match
        the naming convention described in the implementation details section of
        the :mod:`Geometries module <pygfx.geometries>`. If they don't they will
        become optional attributes. Values must either be `Resources` or
        ArrayLike.

    Example
    -------

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
        """Compute the axis-aligned bounding box.

        Computes the aabb based on either positions or the shape of the grid
        buffer. If both are present, the bounding box will be computed based on
        the positions buffer.

        Returns
        -------
        aabb : ndarray, [2, 3]
            The axis-aligned bounding box given by the "smallest" (lowest value)
            and "largest" (highest value) corners.

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
        """Compute a bounding sphere.

        Uses the geometry's axis-aligned bounding box, to estimate a sphere
        which contains the geometry.

        Returns
        -------
        sphere : ndarray, [4]
            A sphere given by it's center and radius. Format: ``(x, y, z, radius)``.

        Notes
        -----
        Since the sphere wraps the geometry's bounding box, it typically won't
        be the minimally binding sphere.

        """
        if self._bsphere is not None and self._bsphere_rev == self._aabb_rev:
            return self._bsphere

        self._bsphere = aabb_to_sphere(self.bounding_box())
        self._bsphere_rev = self._aabb_rev
        return self._bsphere
