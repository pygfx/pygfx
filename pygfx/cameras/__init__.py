""" Objects to view a scene.

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

# flake8: noqa

from ._base import Camera, NDCCamera, ScreenCoordsCamera
from ._perspective import PerspectiveCamera
from ._orthographic import OrthographicCamera

__all__ = [
    "Camera",
    "NDCCamera",
    "ScreenCoordsCamera",
    "PerspectiveCamera",
    "OrthographicCamera",
]
