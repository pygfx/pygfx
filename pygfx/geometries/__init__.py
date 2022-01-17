# flake8: noqa

from ._base import Geometry
from ._box import box_geometry
from ._compat import trimesh_geometry
from ._cylinder import cylinder_geometry, cone_geometry
from ._sphere import sphere_geometry
from ._plane import plane_geometry
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
