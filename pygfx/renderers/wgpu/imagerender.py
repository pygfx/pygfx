import wgpu  # only for flags/enums

from . import register_wgpu_render_function
from ._shadercomposer import Binding, WorldObjectShader
from ._utils import to_texture_format
from ...objects import Image
from ...materials import ImageBasicMaterial
from ...resources import Texture, TextureView


vertex_and_fragment = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT


class BaseImageShader(WorldObjectShader):
    def image_helpers(self):
        return """
        struct ImGeometry {
            indices: array<i32,6>;
            positions: array<vec3<f32>,4>;
            texcoords: array<vec2<f32>,4>;
        };

        fn get_im_geometry() -> ImGeometry {
            let size = textureDimensions(r_tex);
            var geo: ImGeometry;

            geo.indices = array<i32,6>(0, 1, 2,   3, 2, 1);

            let pos1 = vec2<f32>(-0.5);
            let pos2 = vec2<f32>(size.xy) + pos1;
            geo.positions = array<vec3<f32>,4>(
                vec3<f32>(pos2.x, pos1.y, 0.0),
                vec3<f32>(pos2.x, pos2.y, 0.0),
                vec3<f32>(pos1.x, pos1.y, 0.0),
                vec3<f32>(pos1.x, pos2.y, 0.0),
            );

            geo.texcoords = array<vec2<f32>,4>(
                vec2<f32>(1.0, 0.0),
                vec2<f32>(1.0, 1.0),
                vec2<f32>(0.0, 0.0),
                vec2<f32>(0.0, 1.0),
            );

            return geo;
        }

        fn sample(texcoord: vec2<f32>, sizef: vec2<f32>) -> vec4<f32> {
            var color_value: vec4<f32>;

            $$ if texture_format == 'f32'
                color_value = textureSample(r_tex, r_sampler, texcoord.xy);
            $$ else
                let texcoords_u = vec2<i32>(texcoord.xy * sizef.xy);
                color_value = vec4<f32>(textureLoad(r_tex, texcoords_u, 0));
            $$ endif

            $$ if climcorrection
                color_value = vec4<f32>(color_value.rgb {{ climcorrection }}, color_value.a);
            $$ endif
            $$ if texture_nchannels == 1
                color_value = vec4<f32>(color_value.rrr, 1.0);
            $$ elif texture_nchannels == 2
                color_value = vec4<f32>(color_value.rrr, color_value.g);
            $$ endif
            return color_value;
        }
    """


@register_wgpu_render_function(Image, ImageBasicMaterial)
def image_renderer(render_info):
    """Render function capable of rendering images."""

    wobject = render_info.wobject
    geometry = wobject.geometry
    material = wobject.material  # noqa
    shader = ImageShader(render_info, climcorrection=False)

    bindings = {}

    bindings[0] = Binding("u_stdinfo", "buffer/uniform", render_info.stdinfo_uniform)
    bindings[1] = Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer)
    bindings[2] = Binding("u_material", "buffer/uniform", material.uniform_buffer)

    topology = wgpu.PrimitiveTopology.triangle_strip
    n = 4

    # Collect texture and sampler
    if geometry.grid is None:
        raise ValueError("Image.geometry must have a grid (texture).")
    else:
        if isinstance(geometry.grid, TextureView):
            view = geometry.grid
        elif isinstance(geometry.grid, Texture):
            view = geometry.grid.get_view(filter="linear")
        else:
            raise TypeError("Image.geometry.grid must be a Texture or TextureView")
        if view.view_dim.lower() != "2d":
            raise TypeError("Image.geometry.grid must a 2D texture (view)")
        # Sampling type
        fmt = to_texture_format(geometry.grid.format)
        if "norm" in fmt or "float" in fmt:
            shader["texture_format"] = "f32"
            if "unorm" in fmt:
                shader["climcorrection"] = " * 255.0"
            elif "snorm" in fmt:
                shader["climcorrection"] = " * 255.0 - 128.0"
        elif "uint" in fmt:
            shader["texture_format"] = "u32"
        else:
            shader["texture_format"] = "i32"
        # Channels
        shader["texture_nchannels"] = len(fmt) - len(fmt.lstrip("rgba"))

    bindings[3] = Binding("r_sampler", "sampler/filtering", view, "FRAGMENT")
    bindings[4] = Binding("r_tex", "texture/auto", view, vertex_and_fragment)

    # Let the shader generate code for our bindings
    for i, binding in bindings.items():
        shader.define_binding(0, i, binding)

    # Get in what passes this needs rendering
    suggested_render_mask = 3
    if material.opacity >= 1 and shader["texture_nchannels"] in (1, 3):
        suggested_render_mask = 1
    elif material.opacity < 1:
        suggested_render_mask = 2

    # Put it together!
    return [
        {
            "suggested_render_mask": suggested_render_mask,
            "render_shader": shader,
            "primitive_topology": topology,
            "indices": (range(n), range(1)),
            "vertex_buffers": {},
            "bindings0": bindings,
        }
    ]


class ImageShader(BaseImageShader):
    def get_code(self):
        return (
            self.get_definitions()
            + self.common_functions()
            + self.image_helpers()
            + self.vertex_shader()
            + self.fragment_shader()
        )

    def vertex_shader(self):
        return """

        struct VertexInput {
            [[builtin(vertex_index)]] vertex_index : u32;
        };


        [[stage(vertex)]]
        fn vs_main(in: VertexInput) -> Varyings {

            var geo = get_im_geometry();

            // Select what face we're at
            let index = i32(in.vertex_index);
            let i0 = geo.indices[index];

            // Sample position, and convert to world pos, and then to ndc
            let data_pos = vec4<f32>(geo.positions[i0], 1.0);
            let world_pos = u_wobject.world_transform * data_pos;
            let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

            var varyings: Varyings;
            varyings.position = vec4<f32>(ndc_pos);
            varyings.world_pos = vec3<f32>(world_pos.xyz);
            varyings.texcoord = vec2<f32>(geo.texcoords[i0]);
            return varyings;
        }
        """

    def fragment_shader(self):
        return """

        [[stage(fragment)]]
        fn fs_main(varyings: Varyings) -> FragmentOutput {
            let sizef = vec2<f32>(textureDimensions(r_tex));
            let color_value = sample(varyings.texcoord.xy, sizef);
            let albeido = (color_value.rgb - u_material.clim[0]) / (u_material.clim[1] - u_material.clim[0]);

            let final_color = vec4<f32>(albeido, color_value.a * u_material.opacity);

            // Wrap up
            apply_clipping_planes(varyings.world_pos);
            var out = get_fragment_output(varyings.position.z, final_color);

            $$ if write_pick
            // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
            out.pick = (
                pick_pack(u32(u_wobject.id), 20) +
                pick_pack(u32(varyings.texcoord.x * 4194304.0), 22) +
                pick_pack(u32(varyings.texcoord.y * 4194304.0), 22)
            );
            $$ endif

            return out;
        }
        """
