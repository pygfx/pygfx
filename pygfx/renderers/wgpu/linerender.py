import wgpu  # only for flags/enums

from . import register_wgpu_render_function
from ._shadercomposer import BaseShader
from ...utils import array_from_shadertype
from ...resources import Buffer
from ...objects import Line
from ...materials import (
    LineMaterial,
    LineThinMaterial,
    LineThinSegmentMaterial,
    LineSegmentMaterial,
    LineArrowMaterial,
)


# ## Notes
#
# Rendering lines on the GPU is hard! The approach used here uses
# VertexId and storage buffers instead of vertex buffers. That way we
# can create multiple vertices for each point on the line. These
# vertices are positioned such that rendering with triangle_strip
# results in a tick line. More info below.
#
# An alternative (used by e.g. ThreeJS) is to draw the geometry of a
# line segment many times using instancing. Note that some Intel drivers
# limit the number of instances to about 5-10k, which is not much considering
# that lines consisting of millions of points can normally run realtime.
# https://wwwtyro.net/2019/11/18/instanced-lines.html
#
# Another alternative is a geometry shader, but wgpu does not have that.
# But we have compute shaders, which can be used to prepare the triangle-based
# geometry and store it in a buffer, which can be used as a vertex buffer in
# the render pass. The downside of this approach is the memory consumption.
# But a hybrid approach may be viable: the current approach using VertexId,
# but preparing some data in a buffer.

# todo: we can learn about dashing, unfolding and more at http://jcgt.org/published/0002/02/08/paper.pdf


renderer_uniform_type = dict(last_i=("int32",))


@register_wgpu_render_function(Line, LineMaterial)
def line_renderer(wobject, render_info):
    """Render function capable of rendering lines."""

    material = wobject.material
    geometry = wobject.geometry
    shader = LineShader()

    assert isinstance(material, LineMaterial)

    positions1 = geometry.positions

    # With vertex buffers, if a shader input is vec4, and the vbo has
    # Nx2, the z and w element will be zero. This works, because for
    # vertex buffers we provide additional information about the
    # striding of the data.
    # With storage buffers (aka SSBO) we just have some bytes that we
    # read from/write to in the shader. This is more free, but it means
    # that the data in the buffer must match with what the shader
    # expects. In addition to that, there's this thing with vec3's which
    # are padded to 16 bytes. So we either have to require our users
    # to provide Nx4 data, or read them as an array of f32.
    # Anyway, extra check here to make sure the data matches!
    # todo: data.something in here, which means we assume numpy-ish arrays
    if positions1.data.shape[1] != 3:
        raise ValueError(
            "For rendering (thick) lines, the geometry.positions must be Nx3."
        )

    uniform_buffer = Buffer(
        array_from_shadertype(renderer_uniform_type), usage="UNIFORM"
    )
    uniform_buffer.data["last_i"] = positions1.nitems - 1

    shader.define_uniform(0, 0, "u_stdinfo", render_info.stdinfo_uniform.data.dtype)
    shader.define_uniform(0, 1, "u_wobject", wobject.uniform_buffer.data.dtype)
    shader.define_uniform(0, 2, "u_material", material.uniform_buffer.data.dtype)
    shader.define_uniform(0, 3, "u_renderer", uniform_buffer.data.dtype)

    bindings0 = {
        0: ("buffer/uniform", render_info.stdinfo_uniform),
        1: ("buffer/uniform", wobject.uniform_buffer),
        2: ("buffer/uniform", material.uniform_buffer),
        3: ("buffer/uniform", uniform_buffer),
    }

    bindings1 = {
        0: ("buffer/read_only_storage", positions1),
    }

    # Global color or per-vertex?
    shader["per_vertex_color"] = False
    if material.vertex_colors:
        colors1 = geometry.colors
        if colors1.data.shape[1] != 4:
            raise ValueError(
                "For rendering (thick) lines, the geometry.colors must be Nx4."
            )
        bindings1[1] = ("buffer/read_only_storage", colors1)
        shader["per_vertex_color"] = True

    if isinstance(material, LineArrowMaterial):
        shader["line_type"] = "arrow"
        n = (positions1.nitems // 2) * 2 * 4
    elif isinstance(material, LineSegmentMaterial):
        shader["line_type"] = "segment"
        n = (positions1.nitems // 2) * 2 * 5
    else:
        shader["line_type"] = "line"
        n = positions1.nitems * 5

    # Done
    wgsl = shader.generate_wgsl()
    return [
        {
            "vertex_shader": (wgsl, "vs_main"),
            "fragment_shader": (wgsl, "fs_main"),
            "primitive_topology": wgpu.PrimitiveTopology.triangle_strip,
            "indices": (n, 1),
            "bindings0": bindings0,
            "bindings1": bindings1,
            "target": None,  # default
        },
    ]


class LineShader(BaseShader):
    def get_code(self):
        return (
            self.get_definitions()
            + self.more_definitions()
            + self.helpers()
            + self.vertex_shader()
            + self.fragment_shader()
        )

    def more_definitions(self):
        return """

        let PI: f32 = 3.14159265359;

        struct VertexInput {
            [[builtin(vertex_index)]] index : u32;
        };
        struct VertexOutput {
            [[location(0)]] line_width_p: f32;
            [[location(1)]] vec_from_node_p: vec2<f32>;
            [[location(2)]] color: vec4<f32>;
            [[location(3)]] vertex_idx: vec2<f32>;
            [[builtin(position)]] pos: vec4<f32>;
        };

        struct VertexFuncOutput {
            i: i32;
            pos: vec4<f32>;
            line_width_p: f32;
            vec_from_node_p: vec2<f32>;
        };

        struct FragmentOutput {
            [[location(0)]] color: vec4<f32>;
            [[location(1)]] pick: vec4<i32>;
            [[builtin(frag_depth)]] depth : f32;
        };

        [[block]]
        struct BufferI32 {
            data: [[stride(4)]] array<i32>;
        };

        [[block]]
        struct BufferF32 {
            data: [[stride(4)]] array<f32>;
        };

        // Could be useful to be able to pass via an index too :)
        // [[group(1), binding(0)]]
        // var<storage,read> s_indices: BufferI32;

        [[group(1), binding(0)]]
        var<storage,read> s_pos: BufferF32;

        $$ if per_vertex_color
        [[group(1), binding(1)]]
        var<storage,read> s_color: BufferF32;
        $$ endif
        """

    def helpers(self):
        return """

        fn get_point_ndc(index:i32) -> vec4<f32> {
            let raw_pos = vec3<f32>(
                s_pos.data[index], s_pos.data[index + 1], s_pos.data[index + 2]
            );
            let world_pos = u_wobject.world_transform * vec4<f32>(raw_pos.xyz, 1.0);
            var ndc_pos: vec4<f32> = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;
            ndc_pos = ndc_pos / ndc_pos.w;
            return ndc_pos;
        }
        """

    def vertex_shader(self):
        if self.kwargs["line_type"] == "arrow":
            core = self.vertex_shader_arrow()
        elif self.kwargs["line_type"] == "segment":
            core = self.vertex_shader_segment()
        else:
            core = self.vertex_shader_line()

        return (
            core
            + """
        [[stage(vertex)]]
        fn vs_main(in: VertexInput) -> VertexOutput {

            let index = i32(in.index);
            let screen_factor = u_stdinfo.logical_size.xy / 2.0;
            let half_line_width:f32 = u_material.thickness * 0.5;  // in logical pixels
            let l2p:f32 = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;

            let result: VertexFuncOutput = get_vertex_result(index, screen_factor, half_line_width, l2p);

            var out: VertexOutput;
            out.pos = result.pos;
            out.line_width_p = result.line_width_p;
            out.vec_from_node_p = result.vec_from_node_p;
            out.vertex_idx = vec2<f32>(f32(result.i / 10000), f32(result.i % 10000));

            $$ if per_vertex_color
            let i = result.i;
            out.color = vec4<f32>(
                s_color.data[i * 4], s_color.data[i * 4 + 1], s_color.data[i * 4 + 2], s_color.data[i * 4 + 3]
            );
            $$ else:
                out.color = u_material.color;
            $$ endif

            return out;
        }
        """
        )

    def vertex_shader_line(self):
        return """
        fn get_vertex_result(
            index:i32, screen_factor:vec2<f32>, half_line_width:f32, l2p:f32
        ) -> VertexFuncOutput {
            // This vertex shader uses VertexId and storage buffers instead of
            // vertex buffers. It creates 5 vertices for each point on the line.
            // You could do with 4 to get bevel joins, but round joins are so
            // much nicer. The extra vertex is used to cover more fragments at
            // the joins (and caps). In the fragment shader we discard fragments
            // that are "out of range", based on a varying that represents the
            // vector from the node to the vertex.
            //
            // Basic algorithm and definitions:
            //
            // - We read the positions of three nodes, the current, previous, and next.
            // - These are converted to logical pixel screen space.
            // - We define four normal vectors (na, nb, nc, nd) which represent the
            //   vertices. Two for the previous segment, two for the next. One extra
            //   normal/vertex is defined at the join.
            // - These calculations are done for each vertex (yeah, bit of a waste),
            //   we select just one as output.
            //
            //            /  o     node 3
            //           /  /  /
            //          d  /  /
            //   - - - b  /  /     corners rectangles a, b, c, d
            //   o-------o  /      the vertex e is extra
            //   - - - - a c
            //                node 2
            //  node 1
            //
            // Note that at the inside of a join, the normals (b and d above)
            // move past each-other, causing the rectangles of both segments to
            // overlap. We could prevent this, but that would screw up
            // v_vec_from_node_p. Furthermore, consider a thick line with dense
            // nodes jumping all over the place, causing a lot of overlap anyway.
            // In summary, this viz relies on depth testing with "less" (or
            // semi-tranparent lines would break).
            //
            // Possible improvents:
            //
            // - can we do dashes/stipling?
            // - also implement bevel and miter joins
            // - also implement different caps
            // - we can prepare the nodes' screen coordinates in a compute shader.

            let i = index / 5;

            // Sample the current node and it's two neighbours, and convert to NDC
            let npos1 = get_point_ndc(i * 3 - 3);
            let npos2 = get_point_ndc(i * 3 + 0);
            let npos3 = get_point_ndc(i * 3 + 3);

            // Convert to logical screen coordinates, because that's were the lines work
            let ppos1 = (npos1.xy + 1.0) * screen_factor;
            let ppos2 = (npos2.xy + 1.0) * screen_factor;
            let ppos3 = (npos3.xy + 1.0) * screen_factor;

            // Get vectors normal to the line segments
            var v1: vec2<f32> = ppos2.xy - ppos1.xy;
            var v2: vec2<f32> = ppos3.xy - ppos2.xy;
            var na: vec2<f32>;
            var nb: vec2<f32>;
            var nc: vec2<f32>;
            var nd: vec2<f32>;
            var ne: vec2<f32>;

            if (i == 0) {
                // This is the first point on the line: create a cap.
                v1 = v2;
                nc = normalize(vec2<f32>(v2.y, -v2.x));
                nd = -nc;
                na = nd;
                nb = nd - normalize(v2);
                ne = nc - normalize(v2);
            } elseif (i == u_renderer.last_i) {
                // This is the last point on the line: create a cap.
                v2 = v1;
                na = normalize(vec2<f32>(v1.y, -v1.x));
                nb = -na;
                ne = nb + normalize(v1);
                nc = na + normalize(v1);
                nd = na;
            } else {
                // Create a join
                na = normalize(vec2<f32>(v1.y, -v1.x));
                nb = -na;
                nc = normalize(vec2<f32>(v2.y, -v2.x));
                nd = -nc;

                // Determine the angle between two of the normals. If this angle is smaller
                // than zero, the inside of the join is at nb/nd, otherwise it is at na/nc.
                var angle:f32 = atan2(na.y, na.x) - atan2(nc.y, nc.x);
                angle = (angle + PI) % (2.0 * PI) - PI;

                // From the angle we can also calculate the intersection of the lines.
                // We express it in a vector magnifier, and limit it to a factor 2,
                // since when the angle is ~pi, the intersection is near infinity.
                // For a bevel join we can omit ne (or set vec_mag to 1.0).
                // For a miter join we'd need an extra vertex to smoothly transition
                // from a miter to a bevel when the angle is too small.
                // Note that ne becomes inf if v1 == v2, but that's ok, because the
                // triangles in which ne takes part are degenerate for this use-case.
                let vec_mag = 1.0 / max(0.25, cos(0.5 * angle));
                ne = normalize(normalize(v1) - normalize(v2)) * vec_mag;
            }

            // Select the correct vector, note that all except ne are unit.
            var vectors = array<vec2<f32>,5>(na, nb, ne, nc, nd);
            let the_vec = vectors[index % 5] * half_line_width;

            var out : VertexFuncOutput;
            out.i = i;
            out.pos = vec4<f32>((ppos2 + the_vec) / screen_factor - 1.0, npos2.zw);
            out.line_width_p = half_line_width * 2.0 * l2p;
            out.vec_from_node_p = the_vec * l2p;
            return out;
        }
        """

    def vertex_shader_segment(self):
        return """
        fn get_vertex_result(
            index:i32, screen_factor:vec2<f32>, half_line_width:f32, l2p:f32
        ) -> VertexFuncOutput {
            // Similar to the regular line shader, except we only draw segments,
            // using 5 vertices per node. Four for the segments, and 1 to create
            // a degenerate triangle for the space in between. So we only draw
            // caps, no joins.

            let i = index / 5;

            // Sample the current node and either of its neighbours
            let i3 = i + 1 - (i % 2) * 2;  // (i + 1) if i is even else (i - 1)
            let npos2 = get_point_ndc(i * 3);
            let npos3 = get_point_ndc(i3 * 3 );
            // Convert to logical screen coordinates, because that's were the lines work
            let ppos2 = (npos2.xy + 1.0) * screen_factor;
            let ppos3 = (npos3.xy + 1.0) * screen_factor;

            // Get vectors normal to the line segments
            var na: vec2<f32>;
            var nb: vec2<f32>;
            var nc: vec2<f32>;
            var nd: vec2<f32>;

            // Get vectors normal to the line segments
            if ((i % 2) == 0) {
                // A left-cap
                let v = normalize(ppos3.xy - ppos2.xy);
                nc = vec2<f32>(v.y, -v.x);
                nd = -nc;
                na = nc - v;
                nb = nd - v;
            } else {
                // A right cap
                let v = normalize(ppos2.xy - ppos3.xy);
                na = vec2<f32>(v.y, -v.x);
                nb = -na;
                nc = na + v;
                nd = nb + v;
            }

            // Select the correct vector
            // Note the replicated vertices to create degenerate triangles
            var vectors = array<vec2<f32>,10>(na, na, nb, nc, nd, na, nb, nc, nd, nd);
            let the_vec = vectors[index % 10] * half_line_width;

            var out : VertexFuncOutput;
            out.i = i;
            out.pos = vec4<f32>((ppos2 + the_vec) / screen_factor - 1.0, npos2.zw);
            out.line_width_p = half_line_width * 2.0 * l2p;
            out.vec_from_node_p = the_vec * l2p;
            return out;
        }
        """

    def vertex_shader_arrow(self):
        return """
        fn get_vertex_result(
            index:i32, screen_factor:vec2<f32>, half_line_width:f32, l2p:f32
        ) -> VertexFuncOutput {
            // Similar to the normal vertex shader, except we only draw segments,
            // using 3 vertices per node: 6 per segment. 4 for the arrow, and 2
            // to create a degenerate triangle for the space in between. So we
            // only draw caps, no joins.

            let i = index / 3;

            // Sample the current node and either of its neighbours
            let i3 = i + 1 - (i % 2) * 2;  // (i + 1) if i is even else (i - 1)
            let npos2 = get_point_ndc(i * 3);
            let npos3 = get_point_ndc(i3 * 3 );
            // Convert to logical screen coordinates, because that's were the lines work
            let ppos2 = (npos2.xy + 1.0) * screen_factor;
            let ppos3 = (npos3.xy + 1.0) * screen_factor;

            // Get vectors normal to the line segments
            var na: vec2<f32>;
            var nb: vec2<f32>;

            // Get vectors normal to the line segments
            if ((i % 2) == 0) {
                // A left-cap
                let v = ppos3.xy - ppos2.xy;
                na = normalize(vec2<f32>(v.y, -v.x)) * half_line_width;
                nb = v;
            } else {
                // A right cap
                let v = ppos2.xy - ppos3.xy;
                na = -0.75 * v;
                nb = normalize(vec2<f32>(-v.y, v.x)) * half_line_width - v;
            }

            // Select the correct vector
            // Note the replicated vertices to create degenerate triangles
            var vectors = array<vec2<f32>,6>(na, na, nb, na, nb, nb);
            let the_vec = vectors[index % 6];

            var out : VertexFuncOutput;
            out.i = i;
            out.pos = vec4<f32>((ppos2 + the_vec) / screen_factor - 1.0, npos2.zw);
            out.line_width_p = half_line_width * 2.0 * l2p;
            out.vec_from_node_p = vec2<f32>(0.0, 0.0);
            return out;
        }
        """

    def fragment_shader(self):
        return """
        [[stage(fragment)]]
        fn fs_main(in: VertexOutput) -> FragmentOutput {

            // Discard fragments outside of the radius. This is what makes round
            // joins and caps. If we ever want bevel or miter joins, we should
            // change the vertex positions a bit, and drop these lines below.
            let dist_to_node_p = length(in.vec_from_node_p);
            if (dist_to_node_p > in.line_width_p * 0.5) {
                discard;
            }

            // Prep
            var out: FragmentOutput;
            var alpha: f32 = 1.0;

            // Anti-aliasing. Note that because of the discarding above, we cannot
            // use MSAA for aa. But maybe we use another generic approach to aa. We'll see.
            // todo: because of this, our line gets a wee bit thinner, so we have to
            // output ticker lines in the vertex shader!
            let aa_width = 1.2;
            alpha = ((0.5 * in.line_width_p) - abs(dist_to_node_p)) / aa_width;
            alpha = pow(min(1.0, alpha), 2.0);

            // The outer edges with lower alpha for aa are pushed a bit back to avoid artifacts.
            out.depth = in.pos.z + 0.0001 * (0.8 - min(0.8, alpha));

            // Set color
            let color = in.color;
            out.color = vec4<f32>(color.rgb, min(1.0, color.a) * alpha);

            // Set picking info. Yes, the vertex_id interpolates correctly in encoded form.
            let vf: f32 = in.vertex_idx.x * 10000.0 + in.vertex_idx.y;
            let vi = i32(vf + 0.5);
            out.pick = vec4<i32>(u_wobject.id, 0, vi, i32((vf - f32(vi)) * 1048576.0));

            return out;
        }
        """


@register_wgpu_render_function(Line, LineThinSegmentMaterial)
@register_wgpu_render_function(Line, LineThinMaterial)
def thin_line_renderer(wobject, render_info):
    """Render function capable of rendering lines."""

    material = wobject.material
    geometry = wobject.geometry
    shader = ThinLineShader()

    positions1 = geometry.positions

    primitive = wgpu.PrimitiveTopology.line_strip
    if isinstance(material, LineThinSegmentMaterial):
        primitive = wgpu.PrimitiveTopology.line_list

    shader.define_uniform(0, 0, "u_stdinfo", render_info.stdinfo_uniform.data.dtype)
    shader.define_uniform(0, 1, "u_wobject", wobject.uniform_buffer.data.dtype)
    shader.define_uniform(0, 2, "u_material", material.uniform_buffer.data.dtype)

    bindings0 = {
        0: ("buffer/uniform", render_info.stdinfo_uniform),
        1: ("buffer/uniform", wobject.uniform_buffer),
        2: ("buffer/uniform", material.uniform_buffer),
    }

    vertex_buffers = {0: positions1}

    shader["per_vertex_color"] = False
    if material.vertex_colors:
        shader["per_vertex_color"] = True
        colors1 = geometry.colors
        if colors1.data.shape[1] != 4:
            raise ValueError(
                "For rendering (thick) lines, the geometry.colors must be Nx4."
            )
        vertex_buffers[1] = colors1

    wgsl = shader.generate_wgsl()
    return [
        {
            "vertex_shader": (wgsl, "vs_main"),
            "fragment_shader": (wgsl, "fs_main"),
            "primitive_topology": primitive,
            "indices": (positions1.nitems, 1),
            "vertex_buffers": vertex_buffers,
            "bindings0": bindings0,
        },
    ]


class ThinLineShader(BaseShader):
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
            [[location(0)]] pos : vec3<f32>;
            $$ if per_vertex_color
            [[location(1)]] color : vec4<f32>;
            $$ endif
            [[builtin(vertex_index)]] index : u32;
        };
        struct VertexOutput {
            $$ if per_vertex_color
            [[location(0)]] color: vec4<f32>;
            $$ endif
            [[builtin(position)]] pos: vec4<f32>;
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

            let wpos = u_wobject.world_transform * vec4<f32>(in.pos.xyz, 1.0);
            let npos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos;

            var out: VertexOutput;
            out.pos = npos;
            $$ if per_vertex_color
            out.color = in.color;
            $$ endif
            return out;
        }
        """

    def fragment_shader(self):
        return """
        [[stage(fragment)]]
        fn fs_main(in: VertexOutput) -> FragmentOutput {
            var out : FragmentOutput;
            $$ if per_vertex_color
            out.color = in.color;
            $$ else
            out.color = u_material.color;
            $$ endif
            return out;
        }
        """
