from __future__ import annotations

from typing import TYPE_CHECKING, Iterable
import numpy as np

from ..utils.trackable import Store
from ..resources import Resource, Buffer, Texture

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    ArrayOrBuffer = ArrayLike | Buffer
    ArrayOrTexture = ArrayLike | Texture
    ArrayOrTextureOrBuffer = ArrayLike | Texture | Buffer

Optional = None


class Geometry(Store):
    """An object's geometry is a container for data that 'defines' (the shape of) the object.

    This class has no documented properties; attributes (usually buffers and
    sometimes textures) can be freely added to it. What attributes are required
    depends on the kind of object and the usage of the material. Some attributes
    are optional. However, there are several common names.

    Common buffer attributes:

    * ``positions``: an Nx3 buffer representing vertex positions. Used by e.g. ``Mesh``, ``Line``, ``Points``, ``Text``.
    * ``indices``: an Nx3 buffer representing the triangular faces of a ``Mesh``.
    * ``normals``: an Nx3 buffer representing the surface normals of a ``Mesh``.
    * ``texcoords``: an Nx1, Nx2, or Nx3 set of per-vertex texture coordinates. The dimensionality
      should match that of the dimension of the colormap's texture (``material.map``).
    * ``texcoords1``, ``texcoords2`` etc.: for additional texture coordinates. Usually Nx2.
      E.g. a ``TextureMap`` with ``uv_channel`` set to 4 will use "texcoords4".
    * ``colors``: per vertex or per-face color data for e.g. ``Mesh``, ``Line``, ``Points``.
      Can be Nx1 (grayscale), Nx2 (gray plus alpha), Nx3 (RGB), or Nx4 (RGBA).
    * ``sizes``: per vertex sizes for e.g. ``Points``.
    * ``edge_colors`` per vertex edge colors for points with the marker material.
    * ``rotations``: per vertex point/marker rotations.

    Common texture attributes:

    * ``grid``: a 2D or 3D texture for the ``Image`` and ``Volume`` objects, respectively.

    For mesh morphing the following attributes are used:

    * ``morph_targets_relative``: a bool indicating whether the morph positions are relative.
    * ``morph_positions``: a list of arrays of per-vertex morph positions.
    * ``morph_normals``: a list of arrays of per-vertex morph normals.
    * ``morph_colors``: a list of arrays of per-vertex morph colors.

    Instantiation
    -------------
    Most attributes of the geometry are buffers or textures. For convenience, these
    can be passed as arrays, in which case they are automatically wrapped in a buffer
    or texture.

    Example
    -------

    .. code-block:: py

        g = Geometry(positions=[[1, 2], [2, 4], [3, 5], [4, 1]])
        g.positions  # Buffer
        g.positions.data  # numpy array

    """

    def __init__(
        self,
        *,
        positions: ArrayOrBuffer = Optional,
        indices: ArrayOrBuffer = Optional,
        normals: ArrayOrBuffer = Optional,
        texcoords: ArrayOrBuffer = Optional,
        colors: ArrayOrBuffer = Optional,
        grid: ArrayOrTexture = Optional,
        **other_attributes: ArrayOrTextureOrBuffer,
    ):
        super().__init__()

        # We separately declare some possible attribute in the signature to help users via autocompletion.
        # We just merge these with the kwargs so we can process them uniformly.
        all_attributes = other_attributes.copy()
        if positions is not Optional:
            all_attributes["positions"] = positions
        if indices is not Optional:
            all_attributes["indices"] = indices
        if normals is not Optional:
            all_attributes["normals"] = normals
        if texcoords is not Optional:
            all_attributes["texcoords"] = texcoords
        if colors is not Optional:
            all_attributes["colors"] = colors
        if grid is not Optional:
            all_attributes["grid"] = grid

        for name, val in all_attributes.items():
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

    def __repr__(self) -> str:
        # A Store is a subclass of a dict, but it does not look like a dict,
        # e.g. you cannot do geometry.items() or any of the regular dict
        # methods, because *all*  atrribute access is converted to dict key
        # access. So let's forget about this being a dict and also provide a
        # useful repr.
        lines = ["Geometry("]
        for key in dir(self):
            val = self[key]
            lines.append(f"    {key}={val!r},")
        lines.append(f") # at {hex(id(self))}")
        return "\n".join(lines)

    def __dir__(self) -> Iterable[str]:
        return sorted([name for name in self if not name.startswith("_trackable_")])
