import wgpu  # only for flags/enums

from . import register_wgpu_render_function
from ._shadercomposer import Binding, WorldObjectShader
from ...objects import Points
from ...materials import PointsMaterial, GaussianPointsMaterial


@register_wgpu_render_function(Points, PointsMaterial)
def points_renderer(wobject, render_info):
    """Render function capable of rendering Points."""

    geometry = wobject.geometry
    material = wobject.material
    shader = PointsShader(
        wobject, type="circle", per_vertex_sizes=False, per_vertex_colors=False
    )
    n = geometry.positions.nitems * 6

    bindings = {}

    bindings[0] = Binding("u_stdinfo", "buffer/uniform", render_info.stdinfo_uniform)
    bindings[1] = Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer)
    bindings[2] = Binding("u_material", "buffer/uniform", material.uniform_buffer)

    bindings[3] = Binding(
        "s_positions", "buffer/read_only_storage", geometry.positions, "VERTEX"
    )

    if material.vertex_colors:
        bindings[5] = Binding(
            "s_colors", "buffer/read_only_storage", geometry.colors, "VERTEX"
        )
        shader["per_vertex_colors"] = True

    if material.vertex_sizes:
        bindings[4] = Binding(
            "s_sizes", "buffer/read_only_storage", geometry.sizes, "VERTEX"
        )
        shader["per_vertex_sizes"] = True

    if isinstance(material, GaussianPointsMaterial):
        shader["type"] = "gaussian"

    # Let the shader generate code for our bindings
    for i, binding in bindings.items():
        shader.define_binding(0, i, binding)

    # Put it together!
    wgsl = shader.generate_wgsl()
    return [
        {
            "vertex_shader": (wgsl, "vs_main"),
            "fragment_shader": (wgsl, "fs_main"),
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
            + self.more_definitions()
            + self.common_functions()
            + self.vertex_shader()
            + self.fragment_shader()
        )

    def more_definitions(self):
        return """

        struct VertexInput {
            [[builtin(vertex_index)]] vertex_index : u32;
        };
        struct VertexOutput {
            [[location(0)]] pointcoord: vec2<f32>;
            [[location(1)]] vertex_idx: vec2<f32>;
            [[location(2)]] world_pos: vec3<f32>;
            $$ if per_vertex_sizes
            [[location(3)]] size: f32;
            $$ endif
            $$ if per_vertex_colors
            [[location(4)]] color: vec4<f32>;
            $$ endif
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

            $$ if per_vertex_sizes
                let size = load_s_sizes(i0);
                out.size = size;
            $$ else
                let size = u_material.size;
            $$ endif

            let aa_margin = 1.0;
            let delta_logical = deltas[sub_index] * (size + aa_margin);
            let delta_ndc = delta_logical * (1.0 / u_stdinfo.logical_size);
            out.world_pos = world_pos.xyz / world_pos.w;
            out.position = vec4<f32>(ndc_pos.xy + delta_ndc, ndc_pos.zw);
            out.pointcoord = delta_logical;

            $$ if per_vertex_colors
            out.color = load_s_colors(i0);
            $$ endif

            out.vertex_idx = vec2<f32>(f32(i0 / 10000), f32(i0 % 10000));
            return out;
        }
        """

    def fragment_shader(self):
        # Also see See https://github.com/vispy/vispy/blob/master/vispy/visuals/markers.py
        return """

        [[stage(fragment)]]
        fn fs_main(in: VertexOutput) -> FragmentOutput {
            var out: FragmentOutput;

            let d = length(in.pointcoord);
            let aa_width = 1.0;

            $$ if per_vertex_sizes
                let size = in.size;
            $$ else
                let size = u_material.size;
            $$ endif

            $$ if per_vertex_colors
                let color = in.color;
            $$ else
                let color = u_material.color;
            $$ endif

            $$ if type == 'circle'
                if (d <= size - 0.5 * aa_width) {
                    out.color = color;
                } elseif (d <= size + 0.5 * aa_width) {
                    let alpha1 = 0.5 + (size - d) / aa_width;
                    let alpha2 = pow(alpha1, 2.0);  // this works better
                    out.color = vec4<f32>(color.rgb, color.a * alpha2);
                } else {
                    discard;
                }
            $$ elif type == "gaussian"
                if (d <= size) {
                    let sigma = size / 3.0;
                    let t = d / sigma;
                    let a = exp(-0.5 * t * t);
                    out.color = vec4<f32>(color.rgb, color.a * a);
                } else {
                    discard;
                }
            $$ else
                invalid_point_type;
            $$ endif

            // Picking
            let vertex_id = i32(in.vertex_idx.x * 10000.0 + in.vertex_idx.y + 0.5);
            out.pick = vec4<i32>(u_wobject.id, 0, vertex_id, 0);

            out.color.a = out.color.a * u_material.opacity;
            apply_clipping_planes(in.world_pos);
            return out;
        }
        """
