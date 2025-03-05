"""Objects to view a scene.

.. currentmodule:: pygfx.cameras

.. autosummary::
    :toctree: cameras/
    :template: ../_templates/custom_layout.rst

    Camera
    NDCCamera
    ScreenCoordsCamera
    PerspectiveCamera
    OrthographicCamera

"""

# ruff: noqa: F401

from ._base import Camera, NDCCamera, ScreenCoordsCamera
from ._perspective import PerspectiveCamera
from ._orthographic import OrthographicCamera

__all__ = [
    "Camera",
    "NDCCamera",
    "OrthographicCamera",
    "PerspectiveCamera",
    "ScreenCoordsCamera",
]
