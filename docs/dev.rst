Pygfx developer guide
=====================

This document explains the basic for modifying the internals of PyGfx, and for
creating custom materials. See `CONTRIBUTING.md <https://github.com/pygfx/pygfx/blob/main/CONTRIBUTING.md>`_
for basic information on contributing.


Basic concepts
--------------

The central concept of PyGfx is that we have objects organized in a scene graph.
Each object has a geometry that contains the data that defines the object. It also
has a material that defines what the object looks like when it is rendered.

The actual rendering happens by renderers that examine the objects, their geometry and materials, and then do "their magic". In theory the objects and materials are unaware of the different renderers and how they work.
In practice though, PyGfx leans strongly towards GPU rendering and wgpu in specific, and some abstractions leak through (e.g. the materials define uniform buffers, and the ``WorldObject`` has a ``render_mask``.)

Most classes use the ``@property`` decorator to implement properties. Some objects
are "trackable" - they inherit from ``Trackable`` and store some attributes as
``self._store.xx``. These properties are tracked, so that renderers can see when
they have changed and adjust things if needed in an efficient manner.


Library structure
-----------------

The root namespace in general contains everying that the user needs. The
subpackages are organized like this:

* utils: contains utilities used internally as well as public utilities.
  The latter are added to the root namespace.
* resources: contains the ``Buffer``, ``Texture`` and related classes.
* objects: contains all world object classes.
* geometries: contains the ``Geometry`` class and functions to generate specific geometry.
* materials: contains the material classes.
* cameras / controllers / helpers: contains corresponding classes.
* renderers/svg: contains the (very much incomplete) SVG renderer.
* renderers/wgpu: contains the WGPU renderer.


The wgpu renderer
-----------------

The private modules together represent the "engine" itself. This is
probably the most complex part of PyGfx. It is responsible for rendering
the worlds objects in an efficient way, while dealing with the flexible
way in which scene graphs can be changed at runtime. See the docstring in
``__init__.py`` for details.

The public modules all end with "shader" and represent the shaders for
all materials in PyGfx. These are implemented in the same way as youd'd
implement custom materials.


Creating (custom) shaders
-------------------------

A shader object derives from ``WorldObjectShader``. Its purposes is to
provide (templated) WGSL, set the corresponding template variables, define
what bindings (buffers and textures) are used, and provide some more details
of the pipeline and the rendering.

The shader is associated with a WorldObject-material combination using the ``register_wgpu_render_function()``
decorator. This decorator can be applied to your shader class, but it can also
be applied to a function that returns multiple shader objects. This can be useful
if you want multiple "passes", or if you want to prepare date using a custom shader
(e.g. to emulate a geometry shader).

The shader must implement a few methods. A typical shader is shown below:

.. code-block:: python

    from pygfx.renderers.wgpu import (
        register_wgpu_render_function, WorldObjectShader, Binding, RenderMask
    )

    @register_wgpu_render_function(SomeWorldObject, SomeMaterial)
    class SomeShader(WorldObjectShader):

        type = "render"  # must be "render" or "compute"

        def get_resources(self, wobject, shared):
            bindings = {
                0: Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
                ...
            }
            self.define_bindings(0, bindings)
            return {
                "index_buffer": None,
                "vertex_buffers": {},
                "bindings": {
                    0: bindings,
                },
            }

        def get_pipeline_info(self, wobject, shared):
            return {
                "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
                "cull_mode": wgpu.CullMode.none,
            }

        def get_render_info(self, wobject, shared):
            n_vertices = ...
            n_instances = 1
            render_mask = wobject.render_mask
            if not render_mask:
                render_mask = RenderMask.all
            return {
                "indices": (n_vertices, n_instances),
                "render_mask": render_mask,
            }

        def get_code(self):
            return (
                self.code_definitions()
                + self.code_common()
                + self.code_vertex()
                + self.code_fragment()
            )

        def code_vertex(self):
            return """
            @stage(vertex)
            fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
                ...
            }
            """

        def code_fragment(self):
            return """
            @stage(fragment)
            fn fs_main() -> FragmentOutput {
               ...
            }
            """


Remarks:

* In ``get_resources()``, the ``Binding`` object is used to collect all the required information on a binding.
* The wgsl code for a group of bindings can be easily generated using ``define_bindings()``.
* You can also manually define the wgsl code for a binding in cases where this is easier.
  We recommend using another bindgroup for that.
* By convention, methods that return wgsl code are prefixed with "code_".
* The ``render_mask`` specifies in what passes the object must be drawn. Users
  can set it on the object, but by default it is "auto" (0), in which case it must
  be set by the shader. Above it is set to "all" which is a safe option, but if
  the shader knows that all fragments are opaque or all fragments are transparent,
  the ``render_mask`` can be set accordingly.


WGSL code and templating
------------------------

The shader code is written in `WGSL <https://www.w3.org/TR/WGSL/>`_. We use `jinja2-templating <https://jinja.palletsprojects.com/>`_
to allow flexible code generation. Here's an example:

.. code-block:: python

    class SomeShader(WorldObjectShader):

        ...

        def get_resources(self, wobject, shared):
            self["scale"] = 1.2
            ...

        def code_vertex(self):
            return """
            @stage(vertex)
            fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
                let something = x * {{ scale }};
            }
            """

Note that a change to a templating variable requires a recompilation
of the wgpu shader module, which is an expensive operation. Therefore
it's better to use uniforms for things that may change often.


Varyings
--------

Variables passed between vertex shader and fragment shader are called "varyings"
in GPU lingo (because they are interpolated). The ``WorldObjectShader`` has
a system to handle Varyings with relative ease.

...


Outputs and picking
-------------------


Shader library
--------------



Dev reference docs
-------------------

.. autofunction:: pygfx.renderers.wgpu.register_wgpu_render_function

.. autoclass:: pygfx.renderers.wgpu.Binding
    :members:
    :member-order: bysource

.. autoclass:: pygfx.renderers.wgpu.WorldObjectShader
    :members:
    :member-order: bysource

.. autoclass:: pygfx.renderers.wgpu.Shared
    :members:
    :member-order: bysource
