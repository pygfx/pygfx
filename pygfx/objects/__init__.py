""" World Objects and Events.

.. rubric:: Objects
.. autosummary::
    :toctree: objects/
    :template: ../_templates/custom_layout.rst

    WorldObject
    Group
    Scene
    Background
    Points
    Line
    Mesh
    Image
    Volume
    Text
    InstancedMesh
    Light
    PointLight
    DirectionalLight
    AmbientLight
    SpotLight
    LightShadow
    DirectionalLightShadow
    SpotLightShadow
    PointLightShadow

.. currentmodule:: pygfx.objects

.. rubric:: Events
.. autosummary::
    :toctree: objects/
    :template: ../_templates/custom_layout.rst

    Event
    EventTarget
    EventType
    PointerEvent
    KeyboardEvent
    RootEventHandler
    WheelEvent
    WindowEvent

"""

# flake8: noqa

from ._base import WorldObject
from ._events import (
    Event,
    EventTarget,
    EventType,
    PointerEvent,
    KeyboardEvent,
    RootEventHandler,
    WheelEvent,
    WindowEvent,
)
from ._more import Group, Scene, Background, Points, Line, Mesh, Image, Volume, Text
from ._instanced import InstancedMesh
from ._lights import Light, PointLight, DirectionalLight, AmbientLight, SpotLight
from ._lights import (
    LightShadow,
    DirectionalLightShadow,
    SpotLightShadow,
    PointLightShadow,
)
