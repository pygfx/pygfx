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
