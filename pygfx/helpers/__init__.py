"""Helpers for visual debugging of scenes.

This module contains a collection of WorldObjects that can be useful
when debugging a scene or to create reference points within a scene.


.. currentmodule:: pygfx.helpers

.. autosummary::
    :toctree: helpers/
    :template: ../_templates/custom_layout.rst

    AxesHelper
    GridHelper
    BoxHelper
    TransformGizmo
    PointLightHelper
    DirectionalLightHelper
    SpotLightHelper
    Stats

"""

# flake8: noqa

from ._axes import AxesHelper
from ._grid import GridHelper
from ._box import BoxHelper
from ._gizmo import TransformGizmo
from ._lights import (
    PointLightHelper,
    DirectionalLightHelper,
    SpotLightHelper,
)
from ._stats import Stats
