import wgpu  # only for flags/enums

from . import register_wgpu_render_function
from ._shadercomposer import BaseShader
from ...objects import Background
from ...materials import BackgroundMaterial, BackgroundImageMaterial
from ...resources import Texture, TextureView


@register_wgpu_render_function(Background, BackgroundMaterial)
def background_renderer(wobject, render_info):

    material = wobject.material
    shader = BackgroundShader(texture_dim="")

    bindings0 = {
        0: ("buffer/uniform", render_info.stdinfo_uniform),
        1: ("buffer/uniform", wobject.uniform_buffer),
        2: ("buffer/uniform", material.uniform_buffer),
    }

    shader.define_uniform(0, 0, "u_stdinfo", render_info.stdinfo_uniform.data.dtype)
    shader.define_uniform(0, 1, "u_wobject", wobject.uniform_buffer.data.dtype)
    shader.define_uniform(0, 2, "u_material", material.uniform_buffer.data.dtype)

    bindings1 = {}

    if isinstance(material, BackgroundImageMaterial) and material.map is not None:
        if isinstance(material.map, Texture):
            raise TypeError("material.map is a Texture, but must be a TextureView")
        elif not isinstance(material.map, TextureView):
            raise TypeError("material.map must be a TextureView")
        bindings1[0] = "sampler/filtering", material.map
        bindings1[1] = "texture/auto", material.map
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
        if material.map.format.startswith("rgb"):  # rgb maps to rgba
            shader["texture_color"] = True
        elif material.map.format.startswith("r"):
            shader["texture_color"] = False
        else:
            raise ValueError("Unexpected texture format")

    wgsl = shader.generate_wgsl()
    return [
        {
            "vertex_shader": (wgsl, "vs_main"),
            "fragment_shader": (wgsl, "fs_main"),
            "primitive_topology": wgpu.PrimitiveTopology.triangle_strip,
            "indices": 4,
            "bindings0": bindings0,
            "bindings1": bindings1,
        }
    ]


class BackgroundShader(BaseShader):
    def get_code(self):
        return (
            self.get_definitions()
            + self.more_definitions()
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
            [[builtin(position)]] pos: vec4<f32>;
        };

        struct FragmentOutput {
            [[location(0)]] color: vec4<f32>;
            [[location(1)]] pick: vec4<i32>;
        };

        $$ if texture_dim
        [[group(1), binding(0)]]
        var r_sampler: sampler;

        [[group(1), binding(1)]]
        var r_tex: texture_{{ texture_dim }}<f32>;
        $$ endif
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
                // project back to world coords
                let ndc_to_world = u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv;
                //let ndc_to_world = u_stdinfo.ndc_to_world;
                let wpos1_ = ndc_to_world * ndc_pos1;
                let wpos2_ = ndc_to_world * ndc_pos2;
                let wpos1 = wpos1_.xyzw / wpos1_.w;
                let wpos2 = wpos2_.xyzw / wpos2_.w;
                // Store positions and the view direction in the world
                out.pos = ndc_pos1;
                out.texcoord = wpos2.xyz - wpos1.xyz;
            $$ else
                // Store positions and the view direction in the world
                out.pos = vec4<f32>(pos, 0.9999999, 1.0);;
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
                $$ if texture_color
                    out.color = color.rgba;
                $$ else
                    out.color = color.rrra;
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
            return out;
        }
        """
