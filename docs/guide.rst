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


How to use pygfx?
-----------------

Before jumping into the details, here is a minimal example of how to use the
library::

    import pygfx as gfx

    cube = gfx.Mesh(
        gfx.box_geometry(200, 200, 200),
        gfx.MeshPhongMaterial(color="#336699"),
    )

    if __name__ == "__main__":
        gfx.show(cube)

.. image:: _static/guide_hello_world.png

And with that we rendered our first scene using wgpu! Simple, right? At the same
time, this is just scratching the surface of what we can do with pygfx and next
up we will have a look at the three main building blocks involved in creating
more complex rendering setups: (1) `Scenes`, (2) `Canvases`, and (3)
`Renderers`.

**Scenes**

Starting off with the most important building bock, a `Scene` is the world or
scenario to render. It has at least three components: an `object` with some
visual properties, a `Light` source, and a `Camera` to view the scene. Once we
defined those three things, we can position them within our scene and render it. 

Let's look at an example of how this works and recreate the above example. We
begin by defining an empty `Scene`::

    import pygfx as gfx

    scene = gfx.Scene()

Right now this scene is very desolate. There is no light, no objects, and
nothing that can look at those objects. Let's change this by adding some
`light` and creating a `camera`::

    # and god said ...
    scene.add(gfx.AmbientLight())
    scene.add(gfx.DirectionalLight())

    camera = gfx.PerspectiveCamera(70, 16 / 9)
    camera.position.z = 400

Now there is light and a camera to perceive the light. To complete the setup
we also need to add an object to look at::

    geometry = gfx.box_geometry(200, 200, 200)
    material = gfx.MeshPhongMaterial(color="#336699")
    cube = gfx.Mesh(geometry, material)
    scene.add(cube)

Objects are slightly more complicated than lights or cameras. They have a
`geometry`, which controls an object's form, and a `material`, which controls an
object's appearance (color, reflectiveness, etc). From here, we can hand our
result to `gfx.show` and visualize it. This time, however, we are passing a `Scene`
instead of an `Object`, so the result will look a little different::

    gfx.show(scene)

.. image:: _static/guide_static_cube.png

This happens because a complete `Scene` can be rendered as-is, whereas an
`Object` can not. As such, `gfx.show` will, when given an `Object`, create a new
scene for us, add the missing lights, a camera, and a background (for visual
appeal), place the object into the scene and then render the result. When given
a `Scene`, on the other hand, it will use the input as-is, allowing you to see
exactly what you've created and potentially spot any problems.

**Canvases**

The second main building block is the `Canvas`. A `Canvas` provides the surface
onto which the scene should be rendered, and to use it we directly import it
from wgpu-py (on top of which pygfx is built). Wgpu-py has several canvases that
we can choose from, but for starters the most important one is ``auto``, which
automatically selects an appropriate backend to create a window on your screen::

    import pygfx as gfx
    from wgpu.gui.auto import WgpuCanvas

    canvas = WgpuCanvas(size=(200, 200), title="A cube!")
    cube = gfx.Mesh(
        gfx.box_geometry(200, 200, 200),
        gfx.MeshPhongMaterial(color="#336699"),
    )

    if __name__ == "__main__":
        gfx.show(cube, canvas=canvas)

.. image:: _static/guide_hello_world.png

Like before, ``gfx.show`` will automatically create a canvas if we don't provide
one explicitly. This works fine for quick visualizations where the render can
appear as a standalone window. However, if we want to have more fine-graned
control over the target, e.g., because we want to change the window size or
title, we need specify the canvas explicitly. Another common use-case for an
explicit canvas is because we are creating a larger GUI and we want the render
to only appear in a subwidget of the full window.

**Renderers**

The third and final main building block is a `Renderer`. A `Renderer` is like an
artist that brings all of the above together. It looks at the `Scene` through a
`Camera` and draws what it sees onto the surface provided by the `Canvas`. Like
any good artist, a `Renderer` is never seen without its `Canvas`, so to create a
`Renderer` we also need to create a `Canvas`::

    import pygfx as gfx
    from wgpu.gui.auto import WgpuCanvas

    canvas = WgpuCanvas()
    renderer = gfx.renderers.WgpuRenderer(canvas)
    
    cube = gfx.Mesh(
        gfx.box_geometry(200, 200, 200),
        gfx.MeshPhongMaterial(color="#336699"),
    )

    if __name__ == "__main__":
        gfx.show(cube, renderer=renderer)

.. image:: _static/guide_hello_world.png

The output is the same as without the explicit reference because `gfx.show`
will, as you may expect at this point, create a renderer if we don't provide it.
For many applications this is perfectly fine; however, if we want to tackle more
advanced problems (e.g., control the exact process on how objects appear to
overlay each other) we may need to create it explicitly. For starters, it is
enough to know that it exists and what it does, so that we can come back to it
later when it becomes relevant.

Animations
----------

Static renders are nice, but you know what is better? Animations! As mentioned
in the section on `Canvases`, this is done via a backend's event loop which
allows you to specify callbacks that get executed periodically. For convenience,
`gfx.show` exposes two callbacks that will be executed before a new render is
made (`before_render`) and afterward (`after_render`). To animate a scene,
simply pass a callback to this function (here ``animate``) and use it to modify
the scene as desired::

    import pygfx as gfx

    cube = gfx.Mesh(
        gfx.box_geometry(200, 200, 200),
        gfx.MeshPhongMaterial(color="#336699"),
    )

    def animate():
        rot = gfx.linalg.Quaternion().set_from_euler(
                gfx.linalg.Euler(0, 0.01)
            )
        cube.rotation.multiply(rot)

    if __name__ == "__main__":
        gfx.show(cube, before_render=animate)

.. image:: _static/guide_rotating_cube.gif

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


.. _colorspaces:

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

