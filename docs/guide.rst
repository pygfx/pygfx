===============
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
be rendered (we get to the details later).

.. code-block:: python

    scene = gfx.Scene()

    geometry = gfx.BoxGeometry(200, 200, 200)
    material = gfx.MeshPhongMaterial(color=(1, 1, 0, 01))
    cube = gfx.Mesh(geometry, material)

    scene.add(cube)


Opening a window to render to
-----------------------------

Your visualization must end up on the screen somehow. For this, we use the
canvas abstraction provided by wgpu-py.

.. code-block:: python

    # Create Qt widget that can function as a canvas
    from wgpu.gui.qt import WgpuCanvas
    canvas = WgpuCanvas()


We can ask the canvas to schedule a draw event, and tell it what to do
to perform a draw.

.. code-block:: python

    def animate():
       ...  # we'll get to this


    canvas.request_draw(animate)


Setting up a renderer
---------------------

To render your scene to the canvas, you need a renderer. A renderer
needs a target to render to (which can be a canvas or a texture).
And finally, you need a camera to define the point of view to render the scene from.

.. code-block:: python

    # A renderer is associated with a canvas (or a texture) that it renders to
    renderer = gfx.renderers.WgpuRenderer(canvas)

    # A camera defines the viewpoint and projection
    camera = gfx.PerspectiveCamera(70, 16 / 9)

    ...

    # The actual rendering
    renderer.render(scene, camera)


Putting it together
-------------------

If you run this, you should see a rotating yellow cube.

.. code-block:: python

    import pygfx as gfx

    from PySide6 import QtWidgets
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

A world object represents an object in the world. It has a transform, by which the
object can be positioned (translated, rotated, and scaled), and has a visibility property.
These properties apply to the object itself as well as its children (and their children, etc.).


Geometry
--------

Each world object has a geometry. This geometry object contains the
data that defines (the shape of) the object, such as positions, plus
data associated with these positions (normals, texcoords, colors, etc.).
Multiple world objects may share a geometry.


Materials
---------

Each world object also has a material. This material object defines the
appearance of the object. Examples can be its color, how it behaves under lighting,
what render-mode is applied, etc. Multiple world objects may share a material.


Colors
------

Colors in Pygfx can be specified in various ways, e.g.:

.. code-block:: python

    material.color = "red"
    material.color = "#ff0000"
    material.color = 1, 0, 0

Most colors in Pygfx contain four components (including alpha), but can be specified
with 1-4 components:

* a scalar: a grayscale intensity (alpha 1).
* two values: grayscale intensity plus alpha.
* three values: red, green, and blue (i.e. rgb).
* four values: rgb and alpha (i.e. rgba).


Colors for the Mesh, Point, and Line
====================================

These objects can be made a uniform color using `material.color`. More
sophisticated coloring is possible using colormapping and per-vertex
colors.

For Colormapping, the geometry must have a `.texcoords` attribute that
specifies the per-vertex texture coordinates, and the material should
have a `.map` attribute that is a texture in which the final color
will be looked up. The texture can be 1D, 2D or 3D, and the number of columns
in the `geometry.texcoords` should match. This allows for a wide variety of
visualizations.

Per-vertex colors can be specified as `geometry.colors`. They must be enabled
by setting `material.vertex_colors` to `True`.

The colors specified in `material.map` and in `geometry.colors` can have 1-4 values.


Colors in Image and Volume
==========================

The values of the Image and Volume can be either directly interpreted as a color
or can be mapped through a colormap set at `material.map`. If a colormap is used,
it's dimension should match the number of channels in the data. Again,
both direct and colormapped colors can be 1-4 values.


Colorspaces
===========

All colors in PyGfx are interpreted as sRGB by default. This is the same
how webbrowsers interpret colors. Internally, all calculations are performed
in the physical colorspace (sometimes called Linear sRGB) so that these
calculations are physically correct.

If you create a texture with color data that is already in
physical/linear colorspace, you can set the Texture's ``colorspace``
argument to "physical".

Similarly you can use ``Color.from_physical()`` to convert a physical color to sRGB.


Using Pygfx in Jupyter
----------------------

You can use Pygfx in the Jupyter notebook and Jupyter lab. To do so,
use the Jupyter canvas provided by WGPU, and use that canvas as the cell output.

.. code-block:: python

    from wgpu.gui.jupyter import WgpuCanvas

    canvas = WgpuCanvas()
    renderer = gfx.renderers.WgpuRenderer(canvas)

    ...

    canvas  # cell output

Also see the Pygfx examples `here <https://jupyter-rfb.readthedocs.io/en/latest/examples/>`_.

