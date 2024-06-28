"""
Containers for Material data.

.. currentmodule:: pygfx.materials

Materials define how a WorldObject is rendered. Many objects support multiple
different materials, e.g. the materials that can be applied to a Mesh object
mostly determine how the object is affected by lighs. Further, the materials
have properties to influence the rendering, like colors, line thickness,
colormaps, the strength of specular reflections, etc.

.. autosummary::
    :toctree: materials/
    :template: ../_templates/custom_layout.rst

    Material
    material_from_trimesh

    MeshAbstractMaterial
    MeshBasicMaterial
    MeshPhongMaterial
    MeshToonMaterial
    MeshNormalMaterial
    MeshNormalLinesMaterial
    MeshSliceMaterial
    MeshStandardMaterial

    PointsMaterial
    PointsGaussianBlobMaterial
    PointsMarkerMaterial
    PointsSpriteMaterial

    LineMaterial
    LineSegmentMaterial
    LineArrowMaterial
    LineThinMaterial
    LineThinSegmentMaterial
    LineDebugMaterial

    ImageBasicMaterial

    VolumeBasicMaterial
    VolumeIsoMaterial
    VolumeSliceMaterial
    VolumeRayMaterial
    VolumeMipMaterial
    VolumeMinipMaterial

    BackgroundMaterial
    BackgroundImageMaterial
    BackgroundSkyboxMaterial

    GridMaterial

    TextMaterial

"""

# flake8: noqa

from ._base import Material
from ._compat import material_from_trimesh
from ._mesh import (
    MeshAbstractMaterial,
    MeshBasicMaterial,
    MeshPhongMaterial,
    MeshToonMaterial,
    MeshNormalMaterial,
    MeshNormalLinesMaterial,
    MeshSliceMaterial,
    MeshStandardMaterial,
)
from ._points import (
    PointsMaterial,
    PointsGaussianBlobMaterial,
    PointsMarkerMaterial,
    PointsSpriteMaterial,
)
from ._line import (
    LineMaterial,
    LineSegmentMaterial,
    LineArrowMaterial,
    LineThinMaterial,
    LineThinSegmentMaterial,
    LineDebugMaterial,
)
from ._image import ImageBasicMaterial
from ._volume import (
    VolumeBasicMaterial,
    VolumeIsoMaterial,
    VolumeSliceMaterial,
    VolumeRayMaterial,
    VolumeMipMaterial,
    VolumeMinipMaterial,
)
from ._background import (
    BackgroundMaterial,
    BackgroundImageMaterial,
    BackgroundSkyboxMaterial,
)
from ._grid import GridMaterial
from ._text import TextMaterial


# Define __all__ for e.g. Sphinx
__all__ = [
    cls.__name__
    for cls in globals().values()
    if isinstance(cls, type) and issubclass(cls, Material)
]
__all__.sort()
__all__.remove("Material")
__all__.insert(0, "Material")
__all__.extend(["material_from_trimesh"])
