# flake8: noqa

from ._base import Geometry
from ._box import BoxGeometry
from ._cylinder import CylinderGeometry
from ._sphere import SphereGeometry
from ._plane import PlaneGeometry
from ._toroidal import KleinBottleGeometry, TorusKnotGeometry

# Define __all__ for e.g. Sphinx
__all__ = [
    cls.__name__
    for cls in globals().values()
    if isinstance(cls, type) and issubclass(cls, Geometry)
]
__all__.sort()
__all__.remove("Geometry")
__all__.insert(0, "Geometry")
