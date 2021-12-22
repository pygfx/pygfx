"""
Example (and test) for the NDC coordinates. Draws a square that falls partly out of visible range.

* The scene should show a band from the bottom left to the upper right.
* The bottom-left (NDC -1 -1) must be green, the upper-right (NDC 1 1) blue.
* The other corners must be black, cut off at exactly half way: the depth is 0-1.

"""

from wgpu.gui.auto import WgpuCanvas, run
from pygfx.renderers.wgpu._shadercomposer import Binding, WorldObjectShader
import pygfx as gfx


class Square(gfx.WorldObject):
    pass


class SquareMaterial(gfx.Material):
    pass


class SquareShader(WorldObjectShader):
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
        fn vs_main([[builtin(vertex_index)]] index: u32) -> Varyings {
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
        """

    def fragment_shader(self):
        return """
        [[stage(fragment)]]
        fn fs_main(varyings: Varyings) -> FragmentOutput {
            var out: FragmentOutput;
            out.color = varyings.color;
            return out;
        }
        """


@gfx.renderers.wgpu.register_wgpu_render_function(Square, SquareMaterial)
def square_render_function(render_info):
    shader = SquareShader(render_info)
    binding = Binding("u_stdinfo", "buffer/uniform", render_info.stdinfo_uniform)
    shader.define_binding(0, 0, binding)
    return [
        {
            "render_shader": shader,
            "primitive_topology": "triangle-strip",
            "indices": range(4),
            "bindings0": {0: binding},
        },
    ]


# %% Setup scene

canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)

scene = gfx.Scene()
t1 = Square(None, SquareMaterial())
scene.add(t1)

camera = gfx.NDCCamera()  # This example does not even use the camera


if __name__ == "__main__":
    print(__doc__)
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
