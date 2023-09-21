import wgpu  # only for flags/enums

from . import register_wgpu_render_function, WorldObjectShader, Binding, RenderMask
from ._utils import to_texture_format, GfxSampler, GfxTextureView
from ...objects import Image
from ...materials import ImageBasicMaterial
from ...resources import Texture


vertex_and_fragment = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT

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
    def code_image_helpers(self):
        return (
            sampled_value_to_color
            + """
        struct ImGeometry {
            indices: array<i32,6>,
            positions: array<vec3<f32>,4>,
            texcoords: array<vec2<f32>,4>,
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


@register_wgpu_render_function(Image, ImageBasicMaterial)
class ImageShader(BaseImageShader):
    type = "render"

    def get_bindings(self, wobject, shared):
        geometry = wobject.geometry
        material = wobject.material  # noqa

        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
        ]

        self["climcorrection"] = ""

        # Collect texture and sampler
        if geometry.grid is None:
            raise ValueError("Image.geometry must have a grid (texture).")
        else:
            if not isinstance(geometry.grid, Texture):
                raise TypeError("Image.geometry.grid must be a Texture.")
            if geometry.grid.dim != 2:
                raise TypeError("Image.geometry.grid must a 2D texture")
            tex_view = GfxTextureView(geometry.grid)
            sampler = GfxSampler(material.interpolation, "clamp")
            self["colorspace"] = geometry.grid.colorspace
            # Sampling type
            fmt = to_texture_format(geometry.grid.format)
            if "norm" in fmt or "float" in fmt:
                self["img_format"] = "f32"
                if "unorm" in fmt:
                    self["climcorrection"] = " * 255.0"
                elif "snorm" in fmt:
                    self["climcorrection"] = " * 255.0 - 128.0"
            elif "uint" in fmt:
                self["img_format"] = "u32"
            else:
                self["img_format"] = "i32"
            # Channels
            self["img_nchannels"] = len(fmt) - len(fmt.lstrip("rgba"))

        bindings.append(Binding("s_img", "sampler/filtering", sampler, "FRAGMENT"))
        bindings.append(Binding("t_img", "texture/auto", tex_view, vertex_and_fragment))

        # If a colormap is applied ...
        if material.map is not None:
            bindings.extend(
                self.define_img_colormap(material.map, material.map_interpolation)
            )
            self["colorspace"] = material.map.colorspace

        bindings = {i: b for i, b in enumerate(bindings)}
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
        material = wobject.material

        render_mask = wobject.render_mask
        if not render_mask:
            render_mask = RenderMask.all
            if material.is_transparent:
                render_mask = RenderMask.transparent
            elif material.map is not None:
                if self["colormap_nchannels"] in (1, 3):
                    render_mask = RenderMask.opaque
            else:
                if self["img_nchannels"] in (1, 3):
                    render_mask = RenderMask.opaque

        return {
            "indices": (4, 1),
            "render_mask": render_mask,
        }

    def get_code(self):
        return (
            self.code_definitions()
            + self.code_common()
            + self.code_image_helpers()
            + self.code_vertex()
            + self.code_fragment()
        )

    def code_vertex(self):
        return """

        struct VertexInput {
            @builtin(vertex_index) vertex_index : u32,
        };


        @vertex
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

    def code_fragment(self):
        return """

        @fragment
        fn fs_main(varyings: Varyings) -> FragmentOutput {
            let sizef = vec2<f32>(textureDimensions(t_img));
            let value = sample_im(varyings.texcoord.xy, sizef);
            let color = sampled_value_to_color(value);

            // Move to physical colorspace (linear photon count) so we can do math
            $$ if colorspace == 'srgb'
                let physical_color = srgb2physical(color.rgb);
            $$ else
                let physical_color = color.rgb;
            $$ endif
            let opacity = color.a * u_material.opacity;
            let out_color = vec4<f32>(physical_color, opacity);

            // Wrap up
            apply_clipping_planes(varyings.world_pos);
            var out = get_fragment_output(varyings.position.z, out_color);

            $$ if write_pick
            // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
            out.pick = (
                pick_pack(u32(u_wobject.id), 20) +
                pick_pack(u32(varyings.texcoord.x * 4194303.0), 22) +
                pick_pack(u32(varyings.texcoord.y * 4194303.0), 22)
            );
            $$ endif

            return out;
        }
        """
