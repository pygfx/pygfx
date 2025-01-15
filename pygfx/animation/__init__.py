"""
Animation module.

.. currentmodule:: pygfx.animation

This module contains classes for working with animations.

.. autosummary::
    :toctree: animation/
    :template: ../_templates/custom_layout.rst

    LinearInterpolant
    StepInterpolant
    QuaternionLinearInterpolant
    CubicSplineInterpolant
    KeyframeTrack
    AnimationClip

"""

# flake8: noqa

from .interpolant import (
    Interpolant,
    LinearInterpolant,
    StepInterpolant,
    QuaternionLinearInterpolant,
    CubicSplineInterpolant,
)
from .keyframe_track import KeyframeTrack
from .animation_clip import AnimationClip
from .animation_action import AnimationAction
from .animation_mixer import AnimationMixer
from .clock import Clock
