"""
Example that implements a minimal custom object and renders it.

This example simply draws a triangle at the bottomleft of the screen.
It ignores the object's transform and camera, and it does not make use
of geometry or material properties.

It demonstrates:
* How you can define a new WorldObject and Material.
* How to define a render function for it.
* The basic working of the shader class.

"""

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
from pygfx.renderers.wgpu._shadercomposer import Binding, WorldObjectShader


# %% Custom object, material, and matching render function


class Triangle(gfx.WorldObject):
    pass


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
            var positions = array<vec2<f32>, 3>(
                vec2<f32>(10.0, 10.0), vec2<f32>(90.0, 10.0), vec2<f32>(10.0, 90.0)
            );
            let p = 2.0 * positions[index] / u_stdinfo.logical_size - 1.0;
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


@gfx.renderers.wgpu.register_wgpu_render_function(Triangle, TriangleMaterial)
def triangle_render_function(render_info):

    # Create shader object
    shader = TriangleShader(render_info)

    # Define binding, and make shader create code for it
    binding = Binding("u_stdinfo", "buffer/uniform", render_info.stdinfo_uniform)
    shader.define_binding(0, 0, binding)

    # Create dict that the Pygfx renderer needs
    return [
        {
            "render_shader": shader,
            "primitive_topology": "triangle-list",
            "indices": range(3),
            "bindings0": [binding],
        },
    ]


# %% Setup scene

renderer = gfx.WgpuRenderer(WgpuCanvas())
camera = gfx.NDCCamera()  # This material does not actually use the camera

t = Triangle(None, TriangleMaterial())

scene = gfx.Scene()
scene.add(t)


if __name__ == "__main__":
    renderer.request_draw(lambda: renderer.render(scene, camera))
    run()
