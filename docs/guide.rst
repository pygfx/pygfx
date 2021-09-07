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


Defining a scene
----------------

Visualizations in pygfx are constructed of world objects, grouped together into
a scene. These define the kinds of object being visualized, and how it should
be rendered (but we get to details later).

.. code-block::py

    scene = gfx.Scene()

    geometry = gfx.BoxGeometry(200, 200, 200)
    material = gfx.MeshPhongMaterial(color=(1, 1, 0, 01))
    cube = gfx.Mesh(geometry, material)

    scene.add(cube)


Opening a window to render to
-----------------------------

Your visualization must end up on the screen somehow. For this, we use the
canvas abstraction provided by wgpu-py.

.. code-block::py

    # Create Qt widget that can function as a canvas
    from wgpu.gui.qt import WgpuCanvas
    canvas = WgpuCanvas()


We can ask the canvas to schedule a draw event, and tell it what to do
to perform a draw.

.. code-block::py

    def animate():
       ...  # we'll get to this


    canvas.request_draw(animate)


Setting up a renderer
---------------------

To render your scene to the canvas, you need a renderer. And finally,
to specify the angle to look at the scene, you need a camera.

.. code-block::py

    # A renderer is associated with a canvas (or a texture) that it renders to
    renderer = gfx.renderers.WgpuRenderer(canvas)

    # A camera defines the viewpoint in the scene to render from
    camera = gfx.PerspectiveCamera(70, 16 / 9)

    ...

    # The actual rendering
    renderer.render(scene, camera)


Putting it together
-------------------

If you run this, you should see a rotating yellow cube.

.. code-block::py

    import pygfx as gfx

    from PyQt5 import QtWidgets
    from wgpu.gui.qt import WgpuCanvas


    app = QtWidgets.QApplication([])

    # Create a canvas and a renderer
    canvas = WgpuCanvas()
    renderer = gfx.renderers.WgpuRenderer(canvas)

    # Populate a scene with a cube
    scene = gfx.Scene()
    geometry = gfx.BoxGeometry(200, 200, 200)
    material = gfx.MeshPhongMaterial(color=(1, 1, 0, 1))
    cube = gfx.Mesh(geometry, material)
    scene.add(cube)

    camera = gfx.PerspectiveCamera(70, 16 / 9)
    camera.position.z = 400

    def animate():
        rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
        cube.rotation.multiply(rot)

        renderer.render(scene, camera)
        canvas.request_draw()

    canvas.request_draw(animate)
    app.exec_()


World objects
-------------


Geometry
--------


Materials
---------



Using Pygfx in Jupyter
----------------------

You can use Pygfx in the Jupyter notebook and Jupyter lab. To do so,
use the Jupyter canvas provided by WGPU, and use that canvas as the cell output.

.. code-block::py

    from wgpu.gui.jupyter import WgpuCanvas

    canvas = WgpuCanvas()
    renderer = gfx.renderers.WgpuRenderer(canvas)

    ...

    canvas  # cell output

Also see the Pygfx examples `here <https://jupyter-rfb.readthedocs.io/en/latest/examples/>`_.

