"""
Example that implements a simple custom object and renders it.

This example draws a triangle at the appropriate position; the object's
transform and camera are taken into account. It also uses the material
to set the color. But no geometry is used.

It demonstrates:
* How you can define a new WorldObject and Material.
* How to define a render function for it.
* The basic working of the shader class.
* The use of uniforms for material properties.
* The implementation of the camera transforms in the shader.

"""

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
from pygfx.renderers.wgpu._shadercomposer import Binding, WorldObjectShader


# %% Custom object, material, and matching render function


class Triangle(gfx.WorldObject):
    pass


class TriangleMaterial(gfx.Material):

    uniform_type = dict(
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
        self.uniform_buffer.update_range(0, 99999)


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
        fn vs_main([[builtin(vertex_index)]] index: u32) -> Varyings {
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
        """

    def fragment_shader(self):
        return """
        [[stage(fragment)]]
        fn fs_main(varyings: Varyings) -> FragmentOutput {
            var out: FragmentOutput;
            let a = u_material.opacity * u_material.color.a;
            out.color = vec4<f32>(u_material.color.rgb, 1.0);
            return out;
        }
        """


@gfx.renderers.wgpu.register_wgpu_render_function(Triangle, TriangleMaterial)
def triangle_render_function(render_info):
    wobject = render_info.wobject
    material = wobject.material

    # Create shader object
    shader = TriangleShader(render_info)

    # Define bindings, and make shader create code for them
    bindings = [
        Binding("u_stdinfo", "buffer/uniform", render_info.stdinfo_uniform),
        Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
        Binding("u_material", "buffer/uniform", material.uniform_buffer),
    ]
    for i, binding in enumerate(bindings):
        shader.define_binding(0, i, binding)

    # Create dict that the Pygfx renderer needs
    return [
        {
            "render_shader": shader,
            "primitive_topology": "triangle-list",
            "indices": range(3),
            "bindings0": bindings,
        },
    ]


# %% Setup scene

renderer = gfx.WgpuRenderer(WgpuCanvas())
camera = gfx.OrthographicCamera(10, 10)

t = Triangle(None, TriangleMaterial(color="cyan"))
t.position.x = 2  # set offset to demonstrate that it works

scene = gfx.Scene()
scene.add(t)


if __name__ == "__main__":
    renderer.request_draw(lambda: renderer.render(scene, camera))
    run()
