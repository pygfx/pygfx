The pygfx guide
===============


Installation
------------

To install use pip:

.. code-block::

    pip install -U pygfx

or install the bleeding edge from Github:

.. code-block::

    pip install -U https://github.com/pygfx/pygfx/archive/main.zip


What is pygfx?
--------------

PyGfx is a render engine. It renders objects that are organized in a scene, and
provides a way to define what the appearance of these objects should be.
For the actual renderering, multiple render backends are available, but the
main one is based on WGPU.

The `WGPU <https://github.com/pygfx/wgpu-py>`_ library provides the low level API to
communicate with the hardware. WGPU is itself based on Vulkan, Metal and DX12.


Setting up a renderer
---------------------

The result must end up on the screen somehow. For this, we use the
canvas abstraction provided by wgpu-py. Further we need a renderer, a
scene to render, and a camera. These three objects, the canvas,
renderer, and scene must be connected.

.. code-block::

    import pygfx as gfx
    from PyQt5 import QtWidgets
    from wgpu.gui.qt import WgpuCanvas

    app = QtWidgets.QApplication([])

    # Here we create the four main components required for rendering
    canvas = WgpuCanvas()
    renderer = gfx.renderers.WgpuRenderer(canvas)
    scene = gfx.Scene()
    camera = gfx.PerspectiveCamera(70, 16 / 9)

    # We define a function, in which we invoke the renderer, telling
    # it what scene to render and from what viewpoint (the camera).
    def animate():
        renderer.render(scene, camera)

    # Here we ask the canvas to perform a draw soon, and also tell it
    # what to do in all draws from now on.
    canvas.request_draw(animate)


The code above works, but produces a black window. We need objects!


World objects
-------------


Geometry
--------


Materials
---------


Renderers
---------
