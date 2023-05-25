"""
Containers for Material data.

Materials define how a WorldObject is rendered. Many objects support multiple
different materials, e.g. the materials that can be applied to a Mesh object
mostly determine how the object is affected by lighs. Further, the materials
have properties to influence the rendering, like colors, line thickness,
colormaps, the strength of specular reflections, etc.

.. currentmodule:: pygfx.materials

.. autosummary::
    :toctree: materials/
    :template: ../_templates/custom_layout.rst

    Material
    material_from_trimesh

    MeshAbstractMaterial
    MeshBasicMaterial
    MeshPhongMaterial
    MeshNormalMaterial
    MeshNormalLinesMaterial
    MeshSliceMaterial
    MeshStandardMaterial

    PointsMaterial
    GaussianPointsMaterial

    LineMaterial
    LineThinMaterial
    LineThinSegmentMaterial
    LineSegmentMaterial
    LineArrowMaterial

    ImageBasicMaterial

    VolumeBasicMaterial
    VolumeSliceMaterial
    VolumeRayMaterial
    VolumeMipMaterial

    BackgroundMaterial
    BackgroundImageMaterial
    BackgroundSkyboxMaterial

    TextMaterial

"""


# flake8: noqa

from ._base import Material
from ._compat import material_from_trimesh
from ._mesh import (
    MeshAbstractMaterial,
    MeshBasicMaterial,
    MeshPhongMaterial,
    MeshNormalMaterial,
    MeshNormalLinesMaterial,
    MeshSliceMaterial,
    MeshStandardMaterial,
)
from ._points import PointsMaterial, GaussianPointsMaterial
from ._line import (
    LineMaterial,
    LineThinMaterial,
    LineThinSegmentMaterial,
    LineSegmentMaterial,
    LineArrowMaterial,
)
from ._image import ImageBasicMaterial
from ._volume import (
    VolumeBasicMaterial,
    VolumeSliceMaterial,
    VolumeRayMaterial,
    VolumeMipMaterial,
)
from ._background import (
    BackgroundMaterial,
    BackgroundImageMaterial,
    BackgroundSkyboxMaterial,
)
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
