API Reference
=============

.. rubric:: Public API

The primary way of accessing pygfx is by using the members of its top-level namespace.
Currently this includes the following classes, which comprise the public API:

.. autosummary::

    pygfx.objects.WorldObject
    pygfx.utils.color.Color
    pygfx.utils.load.load_scene
    pygfx.utils.show.show
    pygfx.utils.show.Display
    pygfx.utils.viewport.Viewport
    pygfx.utils.text.font_manager
    pygfx.utils.cm
    pygfx.utils.logger


.. from .resources import *
.. from .cameras import *
.. from .controllers import *

.. from .renderers import *



.. rubric:: Sub-Packages

Internally, pygfx is structured into several sub-packages that provide the
functionality exposed in the top-level namespace. At times, you may wish to
search the docs of these sub-packages for additional information. In that case,
you can read more about them here:

.. autosummary::
    :toctree: _autosummary/

    pygfx.cameras
    pygfx.controllers
    pygfx.geometries
    pygfx.helpers
    pygfx.linalg
    pygfx.materials
    pygfx.objects
    pygfx.renderers
    pygfx.resources
    pygfx.utils


Camera objects
--------------

.. autoclass:: pygfx.Camera
    :members:
    :member-order: bysource

.. autoclass:: pygfx.NDCCamera
    :members:
    :member-order: bysource

.. autoclass:: pygfx.ScreenCoordsCamera
    :members:
    :member-order: bysource

.. autoclass:: pygfx.OrthographicCamera
    :members:
    :member-order: bysource

.. autoclass:: pygfx.PerspectiveCamera
    :members:
    :member-order: bysource


.. Controllers
.. -----------

.. .. automodule:: pygfx.controllers
..     :members:
..     :member-order: bysource
