"""
Simple Custom Object
====================

Example that implements a simple custom object and renders it.

This example draws a triangle at the appropriate position; the object's
transform and camera are taken into account. It also uses the material
to set the color. But no geometry is used.

It demonstrates:

* How you can define a new WorldObject and Material.
* How to define a shader for it.
* The use of uniforms for material properties.
* The implementation of the camera transforms in the shader.

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
    uniform_type = dict(
        gfx.Material.uniform_type,
        color="4xf4",
    )

    def __init__(self, *, color="white", **kwargs):
        super().__init__(**kwargs)
        self.color = color

    @property
    def color(self):
        """The uniform color of the triangle."""
        return gfx.Color(self.uniform_buffer.data["color"])

    @color.setter
    def color(self, color):
        self.uniform_buffer.data["color"] = gfx.Color(color)
        self.uniform_buffer.update_full()


@register_wgpu_render_function(Triangle, TriangleMaterial)
class TriangleShader(BaseShader):
    type = "render"

    def get_bindings(self, wobject, shared):
        # We now use three uniform buffers
        bindings = {
            0: Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            1: Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            2: Binding("u_material", "buffer/uniform", wobject.material.uniform_buffer),
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
        return {
            "indices": (3, 1),
            "render_mask": RenderMask.all,  # Good default
        }

    def get_code(self):
        return """
        {$ include 'pygfx.std.wgsl' $}

        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> Varyings {
            // Transform object positition into NDC coords
            let model_pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
            let world_pos = u_wobject.world_transform * model_pos;
            let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

            // List of relative positions, in logical pixels
            var positions = array<vec2<f32>, 3>(
                vec2<f32>(0.0, -20.0), vec2<f32>(-17.0, 15.0), vec2<f32>(17.0, 15.0)
            );

            // Get position for *this* corner
            let screen_factor = u_stdinfo.logical_size.xy / 2.0;
            let screen_pos_ndc = ndc_pos.xy + positions[index] / screen_factor;

            // Set the output
            var varyings: Varyings;
            varyings.position = vec4<f32>(screen_pos_ndc, ndc_pos.zw);
            return varyings;
        }

        @fragment
        fn fs_main(varyings: Varyings) -> FragmentOutput {
            var out: FragmentOutput;
            let a = u_material.color.a * u_material.opacity;
            out.color = vec4<f32>(u_material.color.rgb, a);
            return out;
        }
        """


# Setup scene

renderer = gfx.WgpuRenderer(WgpuCanvas())
camera = gfx.OrthographicCamera(10, 10)

t = Triangle(None, TriangleMaterial(color="cyan"))
t.local.x = 2  # set offset to demonstrate that it works

scene = gfx.Scene()
scene.add(t)


if __name__ == "__main__":
    renderer.request_draw(lambda: renderer.render(scene, camera))
    run()
