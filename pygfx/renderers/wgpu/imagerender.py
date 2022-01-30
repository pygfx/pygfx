import wgpu  # only for flags/enums

from . import register_wgpu_render_function
from ._shadercomposer import Binding, WorldObjectShader
from ._utils import to_texture_format
from ...objects import Image
from ...materials import ImageBasicMaterial
from ...resources import Texture, TextureView


vertex_and_fragment = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT


def handle_colormap(geometry, material, shader):
    if isinstance(material.map, Texture):
        raise TypeError("material.map is a Texture, but must be a TextureView")
    elif not isinstance(material.map, TextureView):
        raise TypeError("material.map must be a TextureView")
    # Dimensionality
    shader["colormap_dim"] = view_dim = material.map.view_dim
    if material.map.view_dim not in ("1d", "2d", "3d"):
        raise ValueError("Unexpected colormap texture dimension")
    # Texture dim matches image channels
    if int(view_dim[0]) != shader["img_nchannels"]:
        raise ValueError(
            f"Image channels {shader['img_nchannels']} does not match material.map {view_dim}"
        )
    # Sampling type
    fmt = to_texture_format(material.map.format)
    if "norm" in fmt or "float" in fmt:
        shader["colormap_format"] = "f32"
    elif "uint" in fmt:
        shader["colormap_format"] = "u32"
    else:
        shader["colormap_format"] = "i32"
    # Channels
    shader["colormap_nchannels"] = len(fmt) - len(fmt.lstrip("rgba"))
    # Return bindinhs
    return [
        Binding("s_colormap", "sampler/filtering", material.map, "FRAGMENT"),
        Binding("t_colormap", "texture/auto", material.map, "FRAGMENT"),
    ]


@register_wgpu_render_function(Image, ImageBasicMaterial)
def image_renderer(render_info):
    """Render function capable of rendering images."""

    wobject = render_info.wobject
    geometry = wobject.geometry
    material = wobject.material  # noqa
    shader = ImageShader(render_info, climcorrection="")

    bindings = [
        Binding("u_stdinfo", "buffer/uniform", render_info.stdinfo_uniform),
        Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
        Binding("u_material", "buffer/uniform", material.uniform_buffer),
    ]

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
            shader["img_format"] = "f32"
            if "unorm" in fmt:
                shader["climcorrection"] = " * 255.0"
            elif "snorm" in fmt:
                shader["climcorrection"] = " * 255.0 - 128.0"
        elif "uint" in fmt:
            shader["img_format"] = "u32"
        else:
            shader["img_format"] = "i32"
        # Channels
        shader["img_nchannels"] = len(fmt) - len(fmt.lstrip("rgba"))

    bindings.append(Binding("s_img", "sampler/filtering", view, "FRAGMENT"))
    bindings.append(Binding("t_img", "texture/auto", view, vertex_and_fragment))

    # If a colormap is applied ...
    if material.map is not None:
        bindings.extend(handle_colormap(geometry, material, shader))

    # Let the shader generate code for our bindings
    for i, binding in enumerate(bindings):
        shader.define_binding(0, i, binding)

    # Get in what passes this needs rendering
    suggested_render_mask = 3
    if material.opacity < 1:
        suggested_render_mask = 2
    if material.map is not None:
        if shader["colormap_nchannels"] in (1, 3):
            suggested_render_mask = 1
    else:
        if shader["img_nchannels"] in (1, 3):
            suggested_render_mask = 1

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


sampled_value_to_color = """
        fn sampled_value_to_color(value_rgba: vec4<f32>) -> vec4<f32> {

            // Make it the correct dimension
            $$ if img_nchannels == 1
                let value_raw = value_rgba.r;
            $$ elif img_nchannels == 2
                let value_raw = value_rgba.rg;
            $$ elif img_nchannels == 3
                let value_raw = value_rgba.rgb;
            $$ else
                let value_raw = value_rgba.rgba;
            $$ endif

            // Apply contrast limits
            let value_cor = value_raw {{ climcorrection }};
            let value_clim = (value_cor - u_material.clim[0]) / (u_material.clim[1] - u_material.clim[0]);

            // Apply colormap or compose final color
            $$ if colormap_dim
                // In the render function we make sure that colormap_dim matches img_nchannels
                let color = sample_colormap(value_clim);
            $$ else
                $$ if img_nchannels == 1
                    let r = value_clim;
                    let color = vec4<f32>(r, r, r, 1.0);
                $$ elif img_nchannels == 2
                    let color = vec4<f32>(value_clim.rrr, value_raw.g);
                $$ elif img_nchannels == 3
                    let color = vec4<f32>(value_clim.rgb, 1.0);
                $$ else
                    let color = vec4<f32>(value_clim.rgb, value_raw.a);
                $$ endif
            $$ endif

            return color;
        }
"""


class BaseImageShader(WorldObjectShader):
    def image_helpers(self):
        return (
            sampled_value_to_color
            + """
        struct ImGeometry {
            indices: array<i32,6>;
            positions: array<vec3<f32>,4>;
            texcoords: array<vec2<f32>,4>;
        };

        fn get_im_geometry() -> ImGeometry {
            let size = textureDimensions(t_img);
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

        fn sample_im(texcoord: vec2<f32>, sizef: vec2<f32>) -> vec4<f32> {
            $$ if img_format == 'f32'
                return textureSample(t_img, s_img, texcoord.xy);
            $$ else
                let texcoords_u = vec2<i32>(texcoord.xy * sizef.xy);
                return vec4<f32>(textureLoad(t_img, texcoords_u, 0));
            $$ endif
        }
    """
        )


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
            let sizef = vec2<f32>(textureDimensions(t_img));
            let value = sample_im(varyings.texcoord.xy, sizef);
            let color = sampled_value_to_color(value);
            let albeido = color.rgb;

            let final_color = vec4<f32>(albeido, color.a * u_material.opacity);

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
