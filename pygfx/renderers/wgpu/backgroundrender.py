import wgpu  # only for flags/enums

from . import register_wgpu_render_function
from ._shadercomposer import Binding, WorldObjectShader
from ._conv import to_texture_format
from ...objects import Background
from ...materials import BackgroundMaterial, BackgroundImageMaterial
from ...resources import Texture, TextureView


@register_wgpu_render_function(Background, BackgroundMaterial)
def background_renderer(wobject, render_info):

    material = wobject.material
    shader = BackgroundShader(wobject, texture_dim="")

    bindings = {}

    # Uniforms
    bindings[0] = Binding("u_stdinfo", "buffer/uniform", render_info.stdinfo_uniform)
    bindings[1] = Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer)
    bindings[2] = Binding("u_material", "buffer/uniform", material.uniform_buffer)

    if isinstance(material, BackgroundImageMaterial) and material.map is not None:
        if isinstance(material.map, Texture):
            raise TypeError("material.map is a Texture, but must be a TextureView")
        elif not isinstance(material.map, TextureView):
            raise TypeError("material.map must be a TextureView")
        bindings[3] = Binding(
            "r_sampler", "sampler/filtering", material.map, "FRAGMENT"
        )
        bindings[4] = Binding("r_tex", "texture/auto", material.map, "FRAGMENT")
        # Select shader
        if material.map.view_dim == "cube":
            shader["texture_dim"] = "cube"
        elif material.map.view_dim == "2d":
            shader["texture_dim"] = "2d"
        else:
            raise ValueError(
                "BackgroundImageMaterial should have map with texture view 2d or cube."
            )
        # Channels
        fmt = to_texture_format(material.map.format)
        shader["texture_nchannels"] = len(fmt) - len(fmt.lstrip("rgba"))

    # Let the shader generate code for our bindings
    for i, binding in bindings.items():
        shader.define_binding(0, i, binding)

    wgsl = shader.generate_wgsl()
    return [
        {
            "vertex_shader": (wgsl, "vs_main"),
            "fragment_shader": (wgsl, "fs_main"),
            "primitive_topology": wgpu.PrimitiveTopology.triangle_strip,
            "indices": 4,
            "bindings0": bindings,
        }
    ]


class BackgroundShader(WorldObjectShader):
    def get_code(self):
        return (
            self.get_definitions()
            + self.more_definitions()
            + self.common_functions()
            + self.vertex_shader()
            + self.fragment_shader()
        )

    def more_definitions(self):
        return """

        struct VertexInput {
            [[builtin(vertex_index)]] index : u32;
        };
        struct VertexOutput {
            [[location(0)]] texcoord: vec3<f32>;
            [[location(1)]] world_pos: vec3<f32>;
            [[builtin(position)]] position: vec4<f32>;
        };

        struct FragmentOutput {
            [[location(0)]] color: vec4<f32>;
            [[location(1)]] pick: vec4<i32>;
        };
    """

    def vertex_shader(self):
        return """
        [[stage(vertex)]]
        fn vs_main(in: VertexInput) -> VertexOutput {
            var out: VertexOutput;
            // Define positions at the four corners of the viewport, at the largest depth
            var positions = array<vec2<f32>, 4>(
                vec2<f32>(-1.0, -1.0),
                vec2<f32>( 1.0, -1.0),
                vec2<f32>(-1.0,  1.0),
                vec2<f32>( 1.0,  1.0),
            );
            // Select the current position
            let pos = positions[i32(in.index)];
            $$ if texture_dim == "cube"
                let ndc_pos1 = vec4<f32>(pos, 0.9999999, 1.0);
                let ndc_pos2 = vec4<f32>(pos, 1.1000000, 1.0);
                let wpos1 = ndc_to_world_pos(ndc_pos1);
                let wpos2 = ndc_to_world_pos(ndc_pos2);
                // Store positions and the view direction in the world
                out.position = ndc_pos1;
                out.world_pos = wpos1;
                out.texcoord = wpos2.xyz - wpos1.xyz;
            $$ else
                // Store positions and the view direction in the world
                out.position = vec4<f32>(pos, 0.9999999, 1.0);
                out.world_pos = ndc_to_world_pos(out.position);
                out.texcoord = vec3<f32>(pos * 0.5 + 0.5, 0.0);
            $$ endif
            return out;
        }
        """

    def fragment_shader(self):
        return """
        [[stage(fragment)]]
        fn fs_main(in: VertexOutput) -> FragmentOutput {
            var out: FragmentOutput;
            $$ if texture_dim
                $$ if texture_dim == '2d'
                    let color = textureSample(r_tex, r_sampler, in.texcoord.xy);
                $$ elif texture_dim == 'cube'
                    let color = textureSample(r_tex, r_sampler, in.texcoord.xyz);
                $$ endif
                $$ if texture_nchannels == 1
                    out.color = vec4<f32>(color.rrr, 1.0);
                $$ elif texture_nchannels == 2
                    out.color = vec4<f32>(color.rrr, color.g);
                $$ else
                    out.color = color;
                $$ endif
            $$ else
                let f = in.texcoord.xy;
                out.color = (
                    u_material.color_bottom_left * (1.0 - f.x) * (1.0 - f.y)
                    + u_material.color_bottom_right * f.x * (1.0 - f.y)
                    + u_material.color_top_left * (1.0 - f.x) * f.y
                    + u_material.color_top_right * f.x * f.y
                );
            $$ endif
            out.color.a = out.color.a * u_material.opacity;
            apply_clipping_planes(in.world_pos);
            return out;
        }
        """
