"""
Example that implements a custom object and renders it.

This example draws multiple triangles. This is more or a full-fledged object.

It demonstrates:
* How you can define a new WorldObject and Material.
* How to define a render function for it.
* The basic working of the shader class.
* The use of uniforms for material properties.
* The implementation of the camera transforms in the shader.
* How geometry (vertex data) can be used in the shader.
* Shader templating.

"""

import numpy as np
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

            let vertex_index = i32(index) / 3;
            let sub_index = i32(index) % 3;

            // Transform object positition into NDC coords
            let model_pos = load_s_positions(vertex_index);  // vec3
            let world_pos = u_wobject.world_transform * vec4<f32>(model_pos, 1.0);
            let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

            // List of relative positions, in logical pixels
            var positions = array<vec2<f32>, 3>(
                vec2<f32>(0.0, -20.0), vec2<f32>(-17.0, 15.0), vec2<f32>(17.0, 15.0)
            );

            // Get position for *this* corner
            let screen_factor = u_stdinfo.logical_size.xy / 2.0;
            let screen_pos_ndc = ndc_pos.xy + {{scale}} * positions[sub_index] / screen_factor;

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


# Tell pygfx to use this render function for a Triangle with TriangleMaterial.
@gfx.renderers.wgpu.register_wgpu_render_function(Triangle, TriangleMaterial)
def triangle_render_function(render_info):
    wobject = render_info.wobject
    material = wobject.material
    geometry = wobject.geometry

    # Create shader object, see the templating variable `scale`
    shader = TriangleShader(render_info, scale=0.5)
    shader.scale = 0.2

    # Define bindings, and make shader create code for them
    bindings = [
        Binding("u_stdinfo", "buffer/uniform", render_info.stdinfo_uniform),
        Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
        Binding("u_material", "buffer/uniform", material.uniform_buffer),
        Binding(
            "s_positions", "buffer/read_only_storage", geometry.positions, "VERTEX"
        ),
    ]
    for i, binding in enumerate(bindings):
        shader.define_binding(0, i, binding)

    # Define in what passes this object is drawn
    # (as a bit mask, 1: opaque, 2: transparent, 3: both).
    if material.opacity < 1:
        suggested_render_mask = 2
    else:
        suggested_render_mask = 1 if material.color[3] >= 1 else 2

    # Create dict that the Pygfx renderer needs
    return [
        {
            "render_shader": shader,
            "primitive_topology": "triangle-list",
            "indices": range(3 * geometry.positions.nitems),
            "bindings0": bindings,
            "suggested_render_mask": suggested_render_mask,
        },
    ]


# %% Setup scene

renderer = gfx.WgpuRenderer(WgpuCanvas())
camera = gfx.OrthographicCamera(10, 10)

t = Triangle(
    gfx.Geometry(positions=np.random.uniform(-4, 4, size=(20, 3)).astype(np.float32)),
    TriangleMaterial(color="yellow"),
)
t.position.x = 2  # set offset to demonstrate that it works

scene = gfx.Scene()
scene.add(t)


if __name__ == "__main__":
    renderer.request_draw(lambda: renderer.render(scene, camera))
    run()
