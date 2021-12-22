"""
Example that implements a custom object and renders it.
"""

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
from pygfx.renderers.wgpu._shadercomposer import Binding, WorldObjectShader


# %% Custom object, material, and matching render function


# This class has mostly a semantic purpose here
class Triangle(gfx.WorldObject):
    pass


# Create a triangle material. We could e.g. define it's color here.
class TriangleMaterial(gfx.Material):
    pass


class TriangleShader(WorldObjectShader):
    def get_code(self):
        return (
            self.get_definitions()
            + self.common_functions()
            + self.vertex_shader()
            + self.fragment_shader()
        )

    def vertex_shader(self):
        return """
        [[stage(vertex)]]
        fn vs_main([[builtin(vertex_index)]] index: u32) -> [[builtin(position)]] vec4<f32> {
            var positions1 = array<vec2<f32>, 3>(vec2<f32>(0.0, -0.5), vec2<f32>(0.5, 0.5), vec2<f32>(-0.5, 0.7));
            var positions2 = array<vec2<f32>, 3>(vec2<f32>(10.0, 10.0), vec2<f32>(90.0, 10.0), vec2<f32>(10.0, 90.0));

            // let p = positions1[index];
            let p = 2.0 * positions2[index] / u_stdinfo.logical_size - 1.0;
            return vec4<f32>(p, 0.0, 1.0);
        }
        """

    def fragment_shader(self):
        return """
        [[stage(fragment)]]
        fn fs_main() -> FragmentOutput {
            var out: FragmentOutput;
            out.color = vec4<f32>(1.0, 0.7, 0.2, 1.0);
            return out;
        }
        """


# Tell pygfx to use this render function for a Triangle with TriangleMaterial.
@gfx.renderers.wgpu.register_wgpu_render_function(Triangle, TriangleMaterial)
def triangle_render_function(render_info):
    shader = TriangleShader(render_info)
    binding = Binding("u_stdinfo", "buffer/uniform", render_info.stdinfo_uniform)
    shader.define_binding(0, 0, binding)
    return [
        {
            "render_shader": shader,
            "primitive_topology": "triangle-list",
            "indices": range(3),
            "bindings0": {0: binding},
        },
    ]


# %% Setup scene

canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)

scene = gfx.Scene()
t1 = Triangle(None, TriangleMaterial())
scene.add(t1)
for _ in range(2):
    scene.add(Triangle(None, TriangleMaterial()))

camera = gfx.NDCCamera()  # This material does not actually use the camera


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
