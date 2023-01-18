"""
Containers for Material data.

Materials define how an object is rendered, subject to certain properties.

.. currentmodule:: pygfx.materials

.. autosummary::
    :toctree: materials/

    Material
    pillow_image
    trimesh_material

    # @almarklein the descriptions for the materials are sparse, but I lack
    # context to come up with something more precise. Could you make a suggestion?
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
from ._compat import pillow_image, trimesh_material
from ._mesh import (
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
__all__.extend(["pillow_image", "trimesh_material"])
