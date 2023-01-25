""" Objects to view a scene.

.. autosummary::
    :toctree: cameras/

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
