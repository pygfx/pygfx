import wgpu  # only for flags/enums

from . import register_wgpu_render_function, WorldObjectShader, Binding, RenderMask
from ._utils import to_texture_format
from ...objects import Background
from ...materials import BackgroundMaterial, BackgroundImageMaterial
from ...resources import Texture, TextureView


@register_wgpu_render_function(Background, BackgroundMaterial)
class BackgroundShader(WorldObjectShader):

    type = "render"

    def get_bindings(self, wobject, shared):
        material = wobject.material

        bindings = {}

        # Uniforms
        bindings[0] = Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer)
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
            # Select texture dimension
            if material.map.view_dim == "cube":
                self["texture_dim"] = "cube"
            elif material.map.view_dim == "2d":
                self["texture_dim"] = "2d"
            else:
                raise ValueError(
                    "BackgroundImageMaterial should have map with texture view 2d or cube."
                )
            # Channels
            fmt = to_texture_format(material.map.format)
            self["texture_nchannels"] = len(fmt) - len(fmt.lstrip("rgba"))
        else:
            self["texture_dim"] = ""

        self.define_bindings(0, bindings)

        return {
            0: bindings,
        }

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_strip,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        return {
            "indices": (4, 1),
            "render_mask": RenderMask.opaque,
        }

    def get_code(self):
        return (
            self.code_definitions()
            + self.code_common()
            + self.code_vertex()
            + self.code_fragment()
        )

    def code_vertex(self):
        return """

        struct VertexInput {
            @builtin(vertex_index) index : u32,
        };

        @stage(vertex)
        fn vs_main(in: VertexInput) -> Varyings {
            var varyings: Varyings;
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
                varyings.position = vec4<f32>(ndc_pos1);
                varyings.world_pos = vec3<f32>(wpos1);
                let d = wpos2.xyz - wpos1.xyz;
                let index = u_material.tex_index.xyz;
                varyings.texcoord = vec3<f32>(d[index.x], -u_material.yscale * d[index.y], d[index.z]);
            $$ else
                // Store positions and the view direction in the world
                varyings.position = vec4<f32>(pos, 0.9999999, 1.0);
                varyings.world_pos = vec3<f32>(ndc_to_world_pos(out.position));
                varyings.texcoord = vec3<f32>(pos * 0.5 + 0.5, 0.0);
            $$ endif
            return varyings;
        }
        """

    def code_fragment(self):
        return """
        @stage(fragment)
        fn fs_main(varyings: Varyings) -> FragmentOutput {
            var final_color : vec4<f32>;
            $$ if texture_dim
                $$ if texture_dim == '2d'
                    let color = textureSample(r_tex, r_sampler, varyings.texcoord.xy);
                $$ elif texture_dim == 'cube'
                    let color = textureSample(r_tex, r_sampler, varyings.texcoord.xyz);
                $$ endif
                $$ if texture_nchannels == 1
                    final_color = vec4<f32>(color.rrr, 1.0);
                $$ elif texture_nchannels == 2
                    final_color = vec4<f32>(color.rrr, color.g);
                $$ else
                    final_color = color;
                $$ endif
            $$ else
                let f = varyings.texcoord.xy;
                final_color = (
                    u_material.color_bottom_left * (1.0 - f.x) * (1.0 - f.y)
                    + u_material.color_bottom_right * f.x * (1.0 - f.y)
                    + u_material.color_top_left * (1.0 - f.x) * f.y
                    + u_material.color_top_right * f.x * f.y
                );
            $$ endif

            // Make physical color with combined alpha
            let physical_color = srgb2physical(final_color.rgb);
            let opacity = final_color.a * u_material.opacity;
            let out_color = vec4<f32>(physical_color, opacity);

            // We can apply clipping planes, but maybe a background should not be clipped?
            // apply_clipping_planes(in.world_pos);

            $$ if not write_pick
                if (true) { discard; }
            $$ endif

            // This is the opaque pass.
            // A fragment of the background could be transparent, but it should still be
            // written in the opaque pass in order for it to really be background.
            // So we fool the blender into thinking this fragment is opaque, even if its not.
            var out = get_fragment_output(varyings.position.z, vec4<f32>(out_color.rgb, 1.0));
            out.color = vec4<f32>(out_color);
            return out;
        }
        """
