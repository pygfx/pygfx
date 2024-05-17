"""Helpers for visual debugging of scenes.

.. currentmodule:: pygfx.helpers

This module contains a collection of WorldObjects that can be useful
when debugging a scene or to create reference points within a scene.

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
    SkeletonHelper

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
from ._skeleton import SkeletonHelper
