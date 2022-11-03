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

Pygfx is structured like many other rendering engines. We create a `scene` (a
worlds/scenarios to render) using three main ingredients: (1) `objects` and
their visual properties, (2) `light` sources, and (3) a `camera` that sees
things. Once we defined those three things, we can position them in our scene
and then use a `renderer` to look at what we have created (or take
pictures/videos of it).

.. note:: 
    The :ref:`full code example <full_example>` can be found below.

Let's look at a hello world example of how this works: A rotating cube. We begin
by defining an empty `Scene`::

    import pygfx as gfx

    scene = gfx.Scene()

Right now this scene is very desolate. There is no light, no objects, and
nothing that can look at those objects. Let's change this by adding some
`light` and a `camera`::

    # and god said ...
    scene.add(gfx.AmbientLight())

    camera = gfx.PerspectiveCamera(70, 16 / 9)
    camera.position.z = 400

Now there is light and a camera to perceive the light. To complete the setup
we also need to add an object to look at::

    geometry = gfx.box_geometry(200, 200, 200)
    material = gfx.MeshPhongMaterial(color=(1, 1, 0, 1))
    cube = gfx.Mesh(geometry, material)

    scene.add(cube)

Objects are slightly more complicated than lights or cameras. They have a
`geometry`, which controlls an object's form, and a `material`, which controls
an object's appearance (color, reflectiveness, etc).

Now we have all the necessary ingredients and it is time to take a look. To do
so we need to create a `canvas` to draw what we see (here an on-screen window)
and a `renderer` that will look at the scene (through the camera) and draw what
it sees onto the canvas::

    from wgpu.gui.auto import WgpuCanvas, run

    renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
    renderer.render(scene, camera)

    run()

.. image:: _static/guide_static_cube.png

Nice, we rendered our first scene! Next-up we can look at a full example that
takes this one step further and animates the cube.

.. _full_example:

Animations
----------

As promised in the previous section, here is a full example of how to use pygfx.
It adds a little bit of flare to the hello world example above  by specifying a
custom `draw_function`. Doing so allows us to add custom logic into the
rendering process, which we can use to animate the cube::

    from wgpu.gui.auto import WgpuCanvas, run

    import pygfx as gfx

    scene = gfx.Scene()

    # create a camera to view the scene
    camera = gfx.PerspectiveCamera(70, 16 / 9)
    camera.position.z = 400

    # add some light so that the camera can see
    scene.add(gfx.AmbientLight())
    scene.add(gfx.DirectionalLight())

    # Populate the scene
    geometry = gfx.box_geometry(200, 200, 200)
    material = gfx.MeshPhongMaterial(color=(1, 1, 0, 1))
    cube = gfx.Mesh(geometry, material)
    scene.add(cube)
    
    # Create a canvas and a renderer
    renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())


    # custom logic to rotate the cube and redraw
    def animate():
        rot = gfx.linalg.Quaternion().set_from_euler(
            gfx.linalg.Euler(0.005, 0.01)
        )
        cube.rotation.multiply(rot)

        renderer.render(scene, camera)
        renderer.request_draw()


    if __name__ == "__main__":
        renderer.request_draw(animate)
        run()

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

