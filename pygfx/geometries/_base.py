from __future__ import annotations

from typing import TYPE_CHECKING, Iterable
import numpy as np
import pylinalg as la

from ..utils.trackable import Trackable
from ..resources import Resource, Buffer, Texture

if TYPE_CHECKING:
    import collections.abc
    from numpy.typing import NDArray, ArrayLike


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

    def __init__(self, **kwargs: Resource | ArrayLike | collections.abc.Buffer):
        super().__init__()

        self._aabb: NDArray | None = None
        self._aabb_rev: int | None = None
        self._bsphere: NDArray | None = None
        self._bsphere_rev: int | None = None

        for name, val in kwargs.items():
            # Get resource object
            if isinstance(val, Resource):
                resource = val
            else:
                # Convert literal arrays to numpy arrays (buffers and textures require memoryview compatible data).
                if isinstance(val, list):
                    dtype = "int32" if name == "indices" else "float32"
                    val = np.array(val, dtype=dtype)
                # Create texture or buffer
                if name == "grid":
                    val = np.asanyarray(val)
                    dim = val.ndim
                    if dim > 2 and val.shape[-1] <= 4:
                        dim -= 1  # last array dim is probably (a subset of) rgba
                    resource = Texture(val, dim=dim)
                else:
                    resource = Buffer(val)

            # Checks
            if isinstance(resource, Buffer):
                format = resource.format
                if name == "indices":
                    # Make no assumptions about shape. Shader will need to validate.
                    # For meshes will be Nx3 or Nx4, but other dtypes may support
                    # multidimensional stuff for fancy graphics.
                    pass
                elif format is None:
                    pass  # buffer with no local data, trust the user
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
                    if not format.startswith(("3x", "4x")):
                        raise ValueError("Expected Nx3 or Nx4 data for colors")
                elif name == "sizes":
                    if not ("x" not in format):
                        raise ValueError("Expected array of scalars for sizes")
                else:
                    pass  # Unknown attribute - no checks

            # Store
            setattr(self, name, resource)

    def __setattr__(self, key: str, new_value: Resource) -> None:
        if not key.startswith(("_", "morph_")):
            if isinstance(new_value, Trackable) or key in self._store:
                return setattr(self._store, key, new_value)
        object.__setattr__(self, key, new_value)

    def __getattribute__(self, key: str) -> Resource:
        if not key.startswith(("_", "morph_")):
            if key in self._store:
                return getattr(self._store, key)
        return object.__getattribute__(self, key)

    def __dir__(self) -> Iterable[str]:
        return [*object.__dir__(self), *self._store]

    def get_bounding_box(self) -> NDArray[np.float32] | None:
        """Compute the axis-aligned bounding box.

        Computes the aabb based on either positions or the shape of the grid
        buffer. If both are present, the bounding box will be computed based on
        the positions buffer.

        Returns
        -------
        aabb : ndarray, [2, 3] or None
            The axis-aligned bounding box given by the "smallest" (lowest value)
            and "largest" (highest value) corners. Is None when the geometry has
            no finite positions.

        """
        if isinstance(positions := getattr(self, "positions", None), Buffer):
            if self._aabb_rev == positions.rev:
                return self._aabb
            aabb: NDArray | None = None
            # Get positions and check expected shape
            pos = positions.data
            if pos.ndim == 2 and pos.shape[1] in (2, 3):
                # Select finite positions
                finite_mask = np.isfinite(pos).all(axis=1)
                if finite_mask.sum() > 0:
                    # Construct aabb
                    pos_finite = pos[finite_mask]
                    aabb = np.array(
                        [pos_finite.min(axis=0), pos_finite.max(axis=0)], "f4"
                    )
                    # If positions contains xy, but not z, assume z=0
                    if aabb.shape[1] == 2:
                        aabb = np.column_stack([aabb, np.zeros((2, 1), "f4")])
            self._aabb = aabb
            self._aabb_rev = self.positions.rev
            return self._aabb

        elif isinstance(grid := getattr(self, "grid", None), Texture):
            if self._aabb_rev == grid.rev:
                return self._aabb
            # account for multi-channel image data
            grid_shape = tuple(reversed(grid.size[: grid.dim]))
            # create aabb in index/data space
            aabb = np.array([np.zeros_like(grid_shape), grid_shape[::-1]], dtype="f8")
            # convert to local image space by aligning
            # center of voxel index (0, 0, 0) with origin (0, 0, 0)
            aabb -= 0.5
            # ensure coordinates are 3D
            # NOTE: important we do this last, we don't want to apply
            # the -0.5 offset to the z-coordinate of 2D images
            if aabb.shape[1] == 2:
                aabb = np.hstack([aabb, [[0], [0]]])
            self._aabb = aabb
            self._aabb_rev = grid.rev
            return self._aabb
        else:
            return None

    def get_bounding_sphere(self) -> NDArray[np.float32] | None:
        """Compute a bounding sphere.

        Uses the geometry's axis-aligned bounding box, to estimate a sphere
        which contains the geometry.

        Returns
        -------
        sphere : ndarray, [4] or None
            A sphere given by it's center and radius. Format: ``(x, y, z, radius)``.
            Is None when the geometry has no finite positions.

        Notes
        -----
        Since the sphere wraps the geometry's bounding box, it typically won't
        be the minimally binding sphere.

        """

        if isinstance(positions := getattr(self, "positions", None), Buffer):
            if self._bsphere_rev == positions.rev:
                return self._bsphere
            bsphere = None
            # Get positions and check expected shape
            pos = positions.data
            if pos.ndim == 2 and pos.shape[1] in (2, 3):
                # Select finite positions
                finite_mask = np.isfinite(pos).all(axis=1)
                if finite_mask.sum() > 0:
                    # Construct aabb
                    pos_finite = pos[finite_mask]
                    center = pos_finite.mean(axis=0)
                    distances = np.linalg.norm(pos_finite - center, axis=0)
                    radius = float(distances.max())
                    if len(center) == 2:
                        bsphere = np.array([center[0], center[1], 0.0, radius], "f4")
                    else:
                        bsphere = np.array(
                            [center[0], center[1], center[2], radius], "f4"
                        )
            self._bsphere = bsphere
            self._bsphere_rev = positions.rev
            return self._bsphere

        else:
            if self._bsphere_rev == self._aabb_rev:
                return self._bsphere
            aabb = self.get_bounding_box()
            self._bsphere = None if aabb is None else la.aabb_to_sphere(aabb)
            self._bsphere_rev = self._aabb_rev
            return self._bsphere
