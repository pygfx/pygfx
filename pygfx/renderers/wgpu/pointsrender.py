import wgpu  # only for flags/enums

from . import register_wgpu_render_function
from ._shadercomposer import Binding, WorldObjectShader
from ._utils import to_vertex_format, to_texture_format
from ...objects import Points
from ...materials import PointsMaterial, GaussianPointsMaterial
from ...resources import Texture, TextureView


def handle_colormap(geometry, material, shader):
    if isinstance(material.map, Texture):
        raise TypeError("material.map is a Texture, but must be a TextureView")
    elif not isinstance(material.map, TextureView):
        raise TypeError("material.map must be a TextureView")
    elif getattr(geometry, "texcoords", None) is None:
        raise ValueError("material.map is present, but geometry has no texcoords")
    # Dimensionality
    shader["colormap_dim"] = view_dim = material.map.view_dim
    if view_dim not in ("1d", "2d", "3d"):
        raise ValueError("Unexpected texture dimension")
    # Texture dim matches texcoords
    vert_fmt = to_vertex_format(geometry.texcoords.format)
    if view_dim == "1d" and "x" not in vert_fmt:
        pass
    elif not vert_fmt.endswith("x" + view_dim[0]):
        raise ValueError(
            f"geometry.texcoords {geometry.texcoords.format} does not match material.map {view_dim}"
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
    # Return bindings
    return [
        Binding("s_colormap", "sampler/filtering", material.map, "FRAGMENT"),
        Binding("t_colormap", "texture/auto", material.map, "FRAGMENT"),
        Binding(
            "s_texcoords", "buffer/read_only_storage", geometry.texcoords, "VERTEX"
        ),
    ]


@register_wgpu_render_function(Points, PointsMaterial)
def points_renderer(render_info):
    """Render function capable of rendering Points."""

    wobject = render_info.wobject
    geometry = wobject.geometry
    material = wobject.material
    shader = PointsShader(
        render_info,
        type="circle",
        per_vertex_sizes=False,
        vertex_color_channels=0,
    )
    n = geometry.positions.nitems * 6

    bindings = [
        Binding("u_stdinfo", "buffer/uniform", render_info.stdinfo_uniform),
        Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
        Binding("u_material", "buffer/uniform", material.uniform_buffer),
        Binding(
            "s_positions", "buffer/read_only_storage", geometry.positions, "VERTEX"
        ),
    ]

    if material.vertex_sizes:
        shader["per_vertex_sizes"] = True
        bindings.append(
            Binding("s_sizes", "buffer/read_only_storage", geometry.sizes, "VERTEX")
        )

    # Per-vertex color, colormap, or a plane color?
    shader["color_mode"] = "uniform"
    if material.vertex_colors:
        shader["color_mode"] = "vertex"
        shader["vertex_color_channels"] = nchannels = geometry.colors.data.shape[1]
        if nchannels not in (1, 2, 3, 4):
            raise ValueError(f"Geometry.colors needs 1-4 columns, not {nchannels}")
        bindings.append(
            Binding("s_colors", "buffer/read_only_storage", geometry.colors, "VERTEX")
        )
    elif material.map is not None:
        shader["color_mode"] = "map"
        bindings.extend(handle_colormap(geometry, material, shader))

    if isinstance(material, GaussianPointsMaterial):
        shader["type"] = "gaussian"

    # Let the shader generate code for our bindings
    for i, binding in enumerate(bindings):
        shader.define_binding(0, i, binding)

    # Determine in what render passes this objects must be rendered
    suggested_render_mask = 3
    if material.opacity < 1:
        suggested_render_mask = 2
    elif shader["color_mode"] == "vertex":
        if shader["vertex_color_channels"] in (1, 3):
            suggested_render_mask = 1
    elif shader["color_mode"] == "map":
        if shader["colormap_nchannels"] in (1, 3):
            suggested_render_mask = 1
    elif shader["color_mode"] == "uniform":
        suggested_render_mask = 1 if material.color[3] >= 1 else 2
    else:
        raise RuntimeError(f"Unexpected color mode {shader['color_mode']}")

    # Put it together!
    return [
        {
            "suggested_render_mask": suggested_render_mask,
            "render_shader": shader,
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "indices": (range(n), range(1)),
            "vertex_buffers": {},
            "bindings0": bindings,
        }
    ]


class PointsShader(WorldObjectShader):

    # Notes:
    # In WGPU, the pointsize attribute can no longer be larger than 1 because
    # of restriction in some hardware/backend API's. So we use our storage-buffer
    # approach (similar for what we use for lines) to sort of fake a geometry shader.
    # An alternative is to use instancing. Could be worth testing both approaches
    # for performance ...

    def get_code(self):
        return (
            self.get_definitions()
            + self.common_functions()
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

            let index = i32(in.vertex_index);
            let i0 = index / 6;
            let sub_index = index % 6;

            let raw_pos = load_s_positions(i0);
            let world_pos = u_wobject.world_transform * vec4<f32>(raw_pos, 1.0);
            let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

            var deltas = array<vec2<f32>, 6>(
                vec2<f32>(-1.0, -1.0),
                vec2<f32>(-1.0,  1.0),
                vec2<f32>( 1.0, -1.0),
                vec2<f32>(-1.0,  1.0),
                vec2<f32>( 1.0, -1.0),
                vec2<f32>( 1.0,  1.0),
            );

            // Need size here in vertex shader too
            $$ if per_vertex_sizes
                let size = load_s_sizes(i0);
            $$ else
                let size = u_material.size;
            $$ endif

            let aa_margin = 1.0;
            let delta_logical = deltas[sub_index] * (size + aa_margin);
            let delta_ndc = delta_logical * (1.0 / u_stdinfo.logical_size);

            var varyings: Varyings;
            varyings.position = vec4<f32>(ndc_pos.xy + delta_ndc, ndc_pos.zw);
            varyings.world_pos = vec3<f32>(world_pos.xyz / world_pos.w);
            varyings.pointcoord = vec2<f32>(delta_logical);
            varyings.size = f32(size);

            // Picking
            varyings.pick_idx = u32(i0);

            // Per-vertex colors
            $$ if vertex_color_channels == 1
            let cvalue = load_s_colors(i0);
            varyings.color = vec4<f32>(cvalue, cvalue, cvalue, 1.0);
            $$ elif vertex_color_channels == 2
            let cvalue = load_s_colors(i0);
            varyings.color = vec4<f32>(cvalue.r, cvalue.r, cvalue.r, cvalue.g);
            $$ elif vertex_color_channels == 3
            varyings.color = vec4<f32>(load_s_colors(i0), 1.0);
            $$ elif vertex_color_channels == 4
            varyings.color = vec4<f32>(load_s_colors(i0));
            $$ endif

            // Set texture coords
            $$ if colormap_dim == '1d'
            varyings.texcoord = f32(load_s_texcoords(i0));
            $$ elif colormap_dim == '2d'
            varyings.texcoord = vec2<f32>(load_s_texcoords(i0));
            $$ elif colormap_dim == '3d'
            varyings.texcoord = vec3<f32>(load_s_texcoords(i0));
            $$ endif

            return varyings;
        }
        """

    def fragment_shader(self):
        # Also see See https://github.com/vispy/vispy/blob/master/vispy/visuals/markers.py
        return """

        [[stage(fragment)]]
        fn fs_main(varyings: Varyings) -> FragmentOutput {
            var final_color : vec4<f32>;

            let d = length(varyings.pointcoord);
            let aa_width = 1.0;

            $$ if per_vertex_sizes
                let size = varyings.size;
            $$ else
                let size = u_material.size;
            $$ endif

            $$ if color_mode == 'vertex'
                let color = varyings.color;
            $$ elif color_mode == 'map'
                let color = sample_colormap(varyings.texcoord);
            $$ else
                let color = u_material.color;
            $$ endif

            $$ if type == 'circle'
                if (d <= size - 0.5 * aa_width) {
                    final_color = color;
                } elseif (d <= size + 0.5 * aa_width) {
                    let alpha1 = 0.5 + (size - d) / aa_width;
                    let alpha2 = pow(alpha1, 2.0);  // this works better
                    final_color = vec4<f32>(color.rgb, color.a * alpha2);
                } else {
                    discard;
                }
            $$ elif type == "gaussian"
                if (d <= size) {
                    let sigma = size / 3.0;
                    let t = d / sigma;
                    let a = exp(-0.5 * t * t);
                    final_color = vec4<f32>(color.rgb, color.a * a);
                } else {
                    discard;
                }
            $$ else
                invalid_point_type;
            $$ endif

            final_color.a = final_color.a * u_material.opacity;

            // Wrap up
            apply_clipping_planes(varyings.world_pos);
            var out = get_fragment_output(varyings.position.z, final_color);

            $$ if write_pick
            // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
            out.pick = (
                pick_pack(u32(u_wobject.id), 20) +
                pick_pack(varyings.pick_idx, 26) +
                pick_pack(u32(varyings.pointcoord.x + 256.0), 9) +
                pick_pack(u32(varyings.pointcoord.y + 256.0), 9)
            );
            $$ endif

            return out;
        }
        """
