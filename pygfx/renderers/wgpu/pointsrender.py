import wgpu  # only for flags/enums

from . import register_wgpu_render_function
from ._shadercomposer import BaseShader
from ...objects import Points
from ...materials import PointsMaterial, GaussianPointsMaterial


@register_wgpu_render_function(Points, PointsMaterial)
def points_renderer(wobject, render_info):
    """Render function capable of rendering meshes displaying a volume slice."""

    geometry = wobject.geometry
    material = wobject.material
    shader = PointsShader(type="circle")
    n = geometry.positions.nitems * 6

    shader.define_uniform(0, 0, "u_stdinfo", render_info.stdinfo_uniform.data.dtype)
    shader.define_uniform(0, 1, "u_wobject", wobject.uniform_buffer.data.dtype)
    shader.define_uniform(0, 2, "u_material", material.uniform_buffer.data.dtype)

    # Collect bindings
    bindings0 = {
        0: ("buffer/uniform", render_info.stdinfo_uniform),
        1: ("buffer/uniform", wobject.uniform_buffer),
        2: ("buffer/uniform", material.uniform_buffer),
    }

    bindings1 = {
        0: ("buffer/read_only_storage", geometry.positions),
    }

    if isinstance(material, GaussianPointsMaterial):
        shader["type"] = "gaussian"

    # Put it together!
    wgsl = shader.generate_wgsl()
    return [
        {
            "vertex_shader": (wgsl, "vs_main"),
            "fragment_shader": (wgsl, "fs_main"),
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "indices": (range(n), range(1)),
            "vertex_buffers": {},
            "bindings0": bindings0,
            "bindings1": bindings1,
        }
    ]


class PointsShader(BaseShader):

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
            + self.vertex_shader()
            + self.fragment_shader()
        )

    def more_definitions(self):
        return """

        struct VertexInput {
            [[builtin(vertex_index)]] index : u32;
        };
        struct VertexOutput {
            [[location(0)]] pointcoord: vec2<f32>;
            [[location(1)]] vertex_idx: vec2<f32>;
            [[builtin(position)]] pos: vec4<f32>;
        };

        struct FragmentOutput {
            [[location(0)]] color: vec4<f32>;
            [[location(1)]] pick: vec4<i32>;
        };

        [[block]]
        struct BufferF32 {
            data: [[stride(4)]] array<f32>;
        };

        [[group(1), binding(0)]]
        var<storage,read> s_pos: BufferF32;
        """

    def vertex_shader(self):
        return """

        [[stage(vertex)]]
        fn vs_main(in: VertexInput) -> VertexOutput {
            var out: VertexOutput;

            let index = i32(in.index);
            let i0 = index / 6;
            let sub_index = index % 6;

            let raw_pos = vec3<f32>(s_pos.data[i0 * 3 + 0], s_pos.data[i0 * 3 + 1], s_pos.data[i0 * 3 + 2]);
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

            let aa_margin = 1.0;
            let delta_logical = deltas[sub_index] * (u_material.size + aa_margin);
            let delta_ndc = delta_logical * (1.0 / u_stdinfo.logical_size);
            out.pos = vec4<f32>(ndc_pos.xy + delta_ndc, ndc_pos.zw);
            out.pointcoord = delta_logical;

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

            let color = u_material.color;
            let d = length(in.pointcoord);
            let size = u_material.size;
            let aa_width = 1.0;

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

            return out;
        }
        """
