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
.. from .objects import *
.. from .geometries import *
.. from .materials import *
.. from .cameras import *
.. from .helpers import *
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


.. Resources
.. ---------

.. .. automodule:: pygfx.resources
..     :members:
..     :member-order: bysource


World objects
-------------

.. autoclass:: pygfx.WorldObject
    :members:
    :member-order: bysource

.. autoclass:: pygfx.Group
    :members:
    :member-order: bysource

.. autoclass:: pygfx.Scene
    :members:
    :member-order: bysource


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


Light objects
--------------

.. autoclass:: pygfx.Light
    :members:
    :member-order: bysource

.. autoclass:: pygfx.AmbientLight
    :members:
    :member-order: bysource

.. autoclass:: pygfx.PointLight
    :members:
    :member-order: bysource

.. autoclass:: pygfx.DirectionalLight
    :members:
    :member-order: bysource

.. autoclass:: pygfx.SpotLight
    :members:
    :member-order: bysource


Specific world objects
----------------------

.. autoclass:: pygfx.Background
    :members:
    :member-order: bysource

.. autoclass:: pygfx.Line
    :members:
    :member-order: bysource

.. autoclass:: pygfx.Mesh
    :members:
    :member-order: bysource

.. autoclass:: pygfx.InstancedMesh
    :members:
    :member-order: bysource

.. autoclass:: pygfx.Points
    :members:
    :member-order: bysource

.. autoclass:: pygfx.Image
    :members:
    :member-order: bysource

.. autoclass:: pygfx.Volume
    :members:
    :member-order: bysource

.. autoclass:: pygfx.Text
    :members:
    :member-order: bysource

.. Controllers
.. -----------

.. .. automodule:: pygfx.controllers
..     :members:
..     :member-order: bysource
