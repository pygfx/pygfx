""" Objects to control cameras.

.. currentmodule:: pygfx.controllers

Controllers define how cameras can be interacted with. They are
independent of any GUI toolkits.

.. autosummary::
    :toctree: controllers/
    :template: ../_templates/custom_layout.rst

    Controller
    PanZoomController
    OrbitController
    TrackballController
    FlyController

"""

# flake8: noqa

from ._base import Controller
from ._panzoom import PanZoomController
from ._orbit import OrbitController
from ._trackball import TrackballController
from ._fly import FlyController


__all__ = [
    "Controller",
    "PanZoomController",
    "OrbitController",
    "TrackballController",
    "FlyController",
]
