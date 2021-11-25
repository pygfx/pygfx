import wgpu  # only for flags/enums

from . import register_wgpu_render_function
from ._shadercomposer import Binding, WorldObjectShader
from ...objects import Points
from ...materials import PointsMaterial, GaussianPointsMaterial


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
        per_vertex_colors=False,
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
    return [
        {
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
            varyings.color = vec4<f32>(load_s_colors(i0));
            varyings.pick_idx = vec2<f32>(f32(i0 / 10000), f32(i0 % 10000));
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

            $$ if per_vertex_colors
                let color = varyings.color;
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
            add_fragment(varyings.position.z, final_color);
            var out = finalize_fragment();

            $$ if pass_index == 0
            let vertex_id = i32(varyings.pick_idx.x * 10000.0 + varyings.pick_idx.y + 0.5);
            out.pick = vec4<i32>(u_wobject.id, 0, vertex_id, 0);
            $$ endif

            return out;
        }
        """
