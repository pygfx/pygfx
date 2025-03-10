"""
Containers for geometry data.

.. currentmodule:: pygfx.geometries

A geometry object contains the data that defines (the shape of) the object, such
as positions, plus data associated with these positions (normals, texcoords,
colors, etc.).

Its attributes are Buffer and Texture(View) objects. The data can be provided as
keyword arguments, which are converted to numpy arrays and wrapped in a Buffer
if necessary.

The names for these attributes are standardized so that the renderers know what
to expect. Each material requires certain attributes to be present, and may
support optional attributes. Optional attributes must always be "turned on" on
the material; their presence on the geometry does not mean that they're used.

The standardized names are:

* ``indices``: An index into per-vertex data. Typically Nx3 for mesh geometry.
* ``positions``: Nx3 positions (xyz), defining the location of e.g. vertices or
  points.
* ``normals``: Nx3 normal vectors. These may or may not be unit.
* ``texcoords``: Texture coordinates used to lookup the color for a vertex.
    Can be Nx1, Nx2 or Nx3, corresponding to a 1D, 2D and 3D texture map.
* ``colors``: Per-vertex colors. Must be NxM, with M 1-4 for gray,
    gray+alpha, rgb, rgba, respectively.
* ``sizes``: Scalar size per-vertex.
* ``grid``: A 2D or 3D Texture that contains a regular grid of data,
  i.e. for images and volumes.

.. rubric:: Basic Geometry
.. autosummary::
    :toctree: geometry/
    :template: ../_templates/custom_layout.rst

    box_geometry
    cylinder_geometry
    cone_geometry
    sphere_geometry
    plane_geometry
    Geometry

.. rubric:: Other/Special Geometry
.. autosummary::
    :toctree: geometry/
    :template: ../_templates/custom_layout.rst

    geometry_from_trimesh
    octahedron_geometry
    icosahedron_geometry
    dodecahedron_geometry
    tetrahedron_geometry
    torus_knot_geometry
    klein_bottle_geometry

"""

# ruff: noqa: F401

from ._base import Geometry
from ._box import box_geometry
from ._compat import geometry_from_trimesh, geometry_from_open3d
from ._cylinder import cylinder_geometry, cone_geometry
from ._sphere import sphere_geometry
from ._plane import plane_geometry, mobius_strip_geometry
from ._polyhedron import (
    octahedron_geometry,
    icosahedron_geometry,
    dodecahedron_geometry,
    tetrahedron_geometry,
)
from ._toroidal import torus_knot_geometry, klein_bottle_geometry

# Define __all__ for e.g. Sphinx
__all__ = [
    ob.__name__
    for ob in globals().values()
    if (isinstance(ob, type) and issubclass(ob, Geometry))
    or (callable(ob) and ob.__name__.endswith("_geometry"))
]
__all__.sort()
__all__.remove("Geometry")
__all__.insert(0, "Geometry")
__all__.append("geometry_from_trimesh")
