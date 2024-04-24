"""
Minimal Custom Object
=====================

Example that implements a minimal custom object and renders it.

This example simply draws a triangle at the bottomleft of the screen.
It ignores the object's transform and camera, and it does not make use
of geometry or material properties.

It demonstrates:

* How you can define a new WorldObject and Material.
* How to define a shader for it.

"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import wgpu
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
from pygfx.renderers.wgpu import (
    Binding,
    BaseShader,
    RenderMask,
    register_wgpu_render_function,
)


# Custom object, material, and matching render function


class Triangle(gfx.WorldObject):
    pass


class TriangleMaterial(gfx.Material):
    pass


@register_wgpu_render_function(Triangle, TriangleMaterial)
class TriangleShader(BaseShader):
    # Mark as render-shader (as opposed to compute-shader)
    type = "render"

    def get_bindings(self, wobject, shared):
        # Our only binding is a uniform buffer
        bindings = {
            0: Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
        }
        self.define_bindings(0, bindings)
        return {
            0: bindings,
        }

    def get_pipeline_info(self, wobject, shared):
        # We draw triangles, no culling
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        # Since we draw only one triangle we need just 3 vertices.
        # Our triangle is opaque (render mask 1).
        return {
            "indices": (3, 1),
            "render_mask": RenderMask.opaque,
        }

    def get_code(self):
        # Here we put together the full (templated) shader code
        return """
        {$ include 'pygfx.std.wgsl' $}

        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
            var positions = array<vec2<f32>, 3>(
                vec2<f32>(10.0, 10.0), vec2<f32>(90.0, 10.0), vec2<f32>(10.0, 90.0)
            );
            let p = 2.0 * positions[index] / u_stdinfo.logical_size - 1.0;
            return vec4<f32>(p, 0.0, 1.0);
        }

        @fragment
        fn fs_main() -> FragmentOutput {
            var out: FragmentOutput;
            out.color = vec4<f32>(1.0, 0.7, 0.2, 1.0);
            return out;
        }
        """


# Setup scene

renderer = gfx.WgpuRenderer(WgpuCanvas())
camera = gfx.NDCCamera()  # This material does not actually use the camera

t = Triangle(None, TriangleMaterial())

scene = gfx.Scene()
scene.add(t)


if __name__ == "__main__":
    renderer.request_draw(lambda: renderer.render(scene, camera))
    run()
