Writing custom shaders
======================

This document explains how to write shaders for Pygfx for the WgpuRenderer.
This may be useful if you want to improve the existing shaders, add new
shaders to Pygfx, or if you want to implement custom shaders in your
own project.


The shader class
----------------

A shader object derives from ``BaseShader``. Its purpose is to
provide (templated) shader-code (WGSL), set the corresponding template variables, define
what bindings (buffers and textures) are used, and provide details
of the pipeline and the rendering.

The shader is associated with a WorldObject-material combination using the ``register_wgpu_render_function()``
decorator. This decorator can be applied to your shader class, but it can also
be applied to a function that returns multiple shader objects. This can be useful
if you want multiple "passes", like a compute pass to prepare some data.

The shader must implement a few methods. A typical shader is shown below:

.. code-block:: python

    from pygfx.renderers.wgpu import (
        register_wgpu_render_function, WorldObjectShader, Binding, RenderMask
    )

    @register_wgpu_render_function(SomeWorldObject, SomeMaterial)
    class SomeShader(WorldObjectShader):

        type = "render"  # must be "render" or "compute"

        def ___init__(self, wobject):
            super().__init__(wobject)

            # The __init__ is a good place to examine the material and geometry and set any template-variables that
            # affect the final wgsl. By accessing `material.has_some_value` here, the value is tracked, so that when
            # `material.has_some_value` changes later, the shader is re-compiled.
            if material.has_some_value:
                self["some_template_variable"] = True

        def get_bindings(self, wobject, shared):

            # You can also set template-variables here. Again, when things that are used here change later, this
            # is detected, and this method will be called again. When a binding has changed (e.g. a colormap is replaced
            # with another) while the formats etc. match, the shader code is not re-composed / re-compiled, making
            # such actions very efficient.
            if getattr(geometry, "colors"):
                self["use_color_buffer"] = True

            # Collect bindings. We must return a dict mapping slot
            # indices to Binding objects. But it's sometimes easier to
            # collect bindings in a list and then convert to a dict.
            bindings = [
                Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
                Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
                Binding("u_material", "buffer/uniform", wobject.material.uniform_buffer),
                ...
            ]
            bindings = {i:b for i, b in enumerate(bindings)}
            # Generate the WGSL code for these bindings
            self.define_bindings(0, bindings)
            # The "bindings" are grouped as a dict of dicts. Often only
            # bind-group 0 is used.
            return {
                0: bindings,
            }

        def get_pipeline_info(self, wobject, shared):
            # Result. All fields are mandatory.
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
            # Result. All fields are mandatory. The RenderMask.all is a safe
            # value; other values are optimizations.
            return {
                "indices": (n_vertices, n_instances),
                "render_mask": render_mask,
            }

        def get_code(self):
            # Return combination of code pieces.
            return """
            {$ include 'pygfx.std.wgsl' $}

            @stage(vertex)
            fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
                ...
            }

            @stage(fragment)
            fn fs_main() -> FragmentOutput {
               ...
            }

            """

Remarks:

* In ``get_bindings()``, the ``Binding`` object is used to collect all the required information on a binding.
* The wgsl code that define a group of bindings is available via ``pygfx.std.wgsl``.
* You can also manually define the wgsl code for a binding in cases where this is easier.
  We recommend using a separate bindgroup for that.
* By convention, methods that return wgsl code are prefixed with "code".
* The ``render_mask`` specifies in what passes the object must be drawn. Users
  can set it on the object, but by default it is "auto" (zero), in which case it must
  be set by the shader. In the code above it is set to "all" which is a safe option, but if
  the shader knows that all fragments are opaque or all fragments are transparent,
  the ``render_mask`` can be set accordingly.


Render passes and render_mask
-----------------------------

When a scene is rendered, it is likely that it's not rendered once, but twice:
one time for the opaque fragments, and one time for the transparent fragments.
This depends on the ``renderer.blend_mode``. It can also be set to just
a single (opaque) pass, or a mode that provides improved handling of transparent
objects that has more than two passes.

Since the used render targets depend on the blend mode and the render
pass, the fragment output is abstracted away for shader authors, as
we'll see further on in this document.

Objects that can have both opaque and transparent fragments, must participate in
all render passes. However, objects that only have opaque fragments or only transparent
fragments, can be optimized. This is what the ``render_mask`` in the previous section
is about. In case of doubt ``RenderMask.all`` is a safe default.


WGSL code and templating
------------------------

The shader code is written in `WGSL <https://www.w3.org/TR/WGSL/>`_. We use `jinja2-templating <https://jinja.palletsprojects.com/>`_
to allow flexible code generation. Here's an example:

.. code-block:: python

        def get_bindings(self, wobject, shared):
            # Template variables can be set like this
            self["scale"] = 1.2
            ...

        def get_code(self):
            # Return combination of code pieces.
            return """
            ...

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
in GPU terminology (because they vary as they are interpolated between
vertices). In Pygfx, each vertex function has a ``Varyings`` as output,
and this is the input of every fragment function. You don't have to
define the ``Varyings`` struct anywhere - Pygfx takes care of that based
on the attributes that are assigned in the vertex shader. The only catch
is that the attributes must be set with an explicit type cast:

.. code-block:: python

        def get_code(self):
            return """
            ...

            @stage(vertex)
            fn vs_main(@builtin(vertex_index) index: u32) -> Varyings {
                ...
                var varyings: Varyings;
                varyings.position = vec4<f32>(screen_pos_ndc, ndc_pos.zw);
                varyings.world_pos = vec3<f32>(world_pos.xyz / world_pos.w);
                return varyings;
            }

            @stage(fragment)
            fn fs_main(varyings: Varyings) -> FragmentOutput {
                ...
                let world_pos = varyings.world_pos;
                ...
            }
            """


FragmentOutput
--------------

In a somewhat similar way, the output of the fragment shader is
predefined. Though in this case the output is determined by the blend
mode and render pass (opaque or transparent), and the details are hidden
from the shader author. This way, Pygfx can support special blend modes
without affecting individual shaders.
All fragment functions in Pygfx look somewhat like this:


.. code-block:: python

        def get_code(self):
            return """
            ...

            @stage(fragment)
            fn fs_main(varyings: Varyings) -> FragmentOutput {
                ...
                var out: FragmentOutput;
                out.color = vec4<f32>(...);
                return out;
            }
            """

For some types of blending the output struct is modified automatically,
and users can influence this process. E.g. to explicitly set a seed for
dithered blending:

.. code-block::

    ...
    var out: FragmentOutput;
    out.color = vec4<f32>(...);
    $$ if blending == 'dither'
    out.seed1 = f32(...);
    $$ endif
    return out;

... or set the weight for weighted blending:

.. code-block::

    ...
    var out: FragmentOutput;
    out.color = vec4<f32>(...);
    $$ if blending == 'weighted'
    out.weight = f32(...);
    $$ endif
    return out;


Picking
-------

The `output` struct of the fragment shader also has a ``pick`` field that can
be set with pointer picking info. To enable picking for a material, use the
``pick_write`` parameter.

.. code-block:: python

    cube = gfx.Mesh(
        gfx.box_geometry(200, 200, 200),
        gfx.MeshBasicMaterial(map=tex, opacity=0.8, pick_write=True),
    )

The picking info returned can vary based on the shader. For all shaders,
it is a ``u64`` into which we can pack as many fields
as needed, using the ``pick_pack()`` function. The material needs to
implement a corresponding ``_wgpu_get_pick_info()`` method
to unpack the picking info. See e.g. the picking of a mesh:

.. code-block:: python

        def get_code(self):
            return """
            ...

            @stage(fragment)
            fn fs_main(varyings: Varyings) -> FragmentOutput {
                ...
                var out: FragmentOutput;
                out.color = color;

                // The builtin write_pick templating variable should be used
                // to ensure picking info is only written in the appropriate render pass
                $$ if write_pick
                // 20 + 26 + 6 + 6 + 6 = 64
                out.pick = (
                    pick_pack(varyings.pick_id, 20) +
                    pick_pack(varyings.pick_idx, 26) +
                    pick_pack(u32(varyings.pick_coords.x * 64.0), 6) +
                    pick_pack(u32(varyings.pick_coords.y * 64.0), 6) +
                    pick_pack(u32(varyings.pick_coords.z * 64.0), 6)
                );
                $$ endif

                return out;
            }
            """


Clipping planes
---------------

For common features that apply to all/most objects, wgsl convenience shader chunks are provided.
included in the shader code using the ``include`` directive. For example, to use clipping planes,
you can include the wgsl code for clipping planes in your shader like this:

.. code-block:: python

        def get_code(self):
            return """
            ...

            @stage(fragment)
            fn fs_main(varyings: Varyings) -> FragmentOutput {
                ...

                // clipping planes
                {$ include 'pygfx.clipping_planes.wgsl' $}

                var out: FragmentOutput;
                out.color = color;
                return out;
            }
            """


Colormapping
------------

Many materials in Pygfx support colormapping. We distinguish between colormaps
with image input data, and vertex input data (texture coordinates). The number of
channels of the input data must match the dimensionality of the colormap (1D, 2D or 3D).

The base shader class has two corresponding helper functions, and there
is a wgsl helper function.

For images / volumes:

.. code-block:: python

        def get_bindings(self, wobjwect, shared):
            ...
            extra_bindings = self.define_img_colormap(material.map)
            bindings.extend(extra_bindings)
            ...

        def get_code(self):
            return """
            {$ include 'pygfx.std.wgsl' $}
            {$ include 'pygfx.colormap.wgsl '$}
            ...

            @stage(fragment)
            fn fs_main(varyings: Varyings) -> FragmentOutput {
                ...
                let img_value = textureSample(t_img, s_img, texcoord.xy);
                let color = sample_colormap(img_value);
                ...
            }
            """

For points / lines, meshes, etc.:

.. code-block:: python

        def get_bindings(self, wobjwect, shared):
            ...
            extra_bindings = self.define_vertex_colormap(material.map, geometry.texcoords)
            bindings.extend(extra_bindings)
            ...

        def get_code(self):
            return """
            {$ include 'pygfx.std.wgsl' $}
            {$ include 'pygfx.colormap.wgsl '$}

            ...
            @stage(fragment)
            fn fs_main(varyings: Varyings) -> FragmentOutput {
                ...
                let color = sample_colormap(varyings.texcoord);
                ...
            }
            """


Lights and shadows
------------------

TODO


Other functions
---------------

Other function that can be used in wgsl are:

* ``ndc_to_world_pos(vec4<f32>) -> vec3<f32>``
