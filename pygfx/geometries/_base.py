from __future__ import annotations

from typing import TYPE_CHECKING, Iterable
import numpy as np

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
