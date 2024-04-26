"""
NDC Coordinates
===============

Example (and test) for the NDC coordinates. Draws a square that falls partly out of visible range.

* The scene should show a band from the bottom left to the upper right.
* The bottom-left (NDC -1 -1) must be green, the upper-right (NDC 1 1) blue.
* The other corners must be black, cut off at exactly half way: the depth is 0-1.

"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

from wgpu.gui.auto import WgpuCanvas, run
from pygfx.renderers.wgpu import (
    Binding,
    BaseShader,
    register_wgpu_render_function,
)
import pygfx as gfx


class Square(gfx.WorldObject):
    pass


class SquareMaterial(gfx.Material):
    pass


@register_wgpu_render_function(Square, SquareMaterial)
class SquareShader(BaseShader):
    def get_bindings(self, wobject, shared):
        binding = Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer)
        self.define_binding(0, 0, binding)
        return {
            0: {0: binding},
        }

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": "triangle-strip",
            "cull_mode": 0,
        }

    def get_render_info(self, wobject, shared):
        return {
            "indices": (4, 1),
            "render_mask": 3,
        }

    def get_code(self):
        return """
        {$ include 'pygfx.std.wgsl' $}

        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> Varyings {
            var positions = array<vec3<f32>, 4>(
                vec3<f32>(-1.0, -1.0, 0.5), vec3<f32>(-1.0, 1.0, 1.5), vec3<f32>(1.0, -1.0, -0.5), vec3<f32>(1.0, 1.0, 0.5)
            );
            var colors = array<vec3<f32>, 4>(
                vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.5, 0.5), vec3<f32>(0.0, 0.5, 0.5), vec3<f32>(0.0, 0.0, 1.0)
            );

            var varyings: Varyings;
            varyings.position = vec4<f32>(positions[index], 1.0);
            varyings.color = vec4<f32>(colors[index], 1.0);
            return varyings;
        }

        @fragment
        fn fs_main(varyings: Varyings) -> FragmentOutput {
            var out: FragmentOutput;
            out.color = varyings.color;
            return out;
        }
        """


# Setup scene

canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)

scene = gfx.Scene()
t1 = Square(None, SquareMaterial())
scene.add(t1)

camera = gfx.NDCCamera()  # This example does not even use the camera

canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    run()
