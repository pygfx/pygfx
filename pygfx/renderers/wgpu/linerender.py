import wgpu  # only for flags/enums

from . import register_wgpu_render_function
from ._shadercomposer import Binding, WorldObjectShader
from .pointsrender import handle_colormap
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


renderer_uniform_type = dict(last_i="i4")


@register_wgpu_render_function(Line, LineMaterial)
def line_renderer(render_info):
    """Render function capable of rendering lines."""

    wobject = render_info.wobject
    material = wobject.material
    geometry = wobject.geometry
    shader = LineShader(
        render_info,
        vertex_color_channels=0,
    )

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

    uniform_buffer = Buffer(array_from_shadertype(renderer_uniform_type))
    uniform_buffer.data["last_i"] = positions1.nitems - 1

    bindings = [
        Binding("u_stdinfo", "buffer/uniform", render_info.stdinfo_uniform),
        Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
        Binding("u_material", "buffer/uniform", material.uniform_buffer),
        Binding("u_renderer", "buffer/uniform", uniform_buffer),
        Binding("s_positions", "buffer/read_only_storage", positions1, "VERTEX"),
    ]

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

    # Triage over materials
    if isinstance(material, LineArrowMaterial):
        shader["line_type"] = "arrow"
        n = (positions1.nitems // 2) * 2 * 4
    elif isinstance(material, LineSegmentMaterial):
        shader["line_type"] = "segment"
        n = (positions1.nitems // 2) * 2 * 5
    else:
        shader["line_type"] = "line"
        n = positions1.nitems * 5

    # Let the shader generate code for our bindings
    for i, binding in enumerate(bindings):
        shader.define_binding(0, i, binding)

    # Determine in what render passes this objects must be rendered
    # Note: the renderer use alpha for aa, so we are never opaque only.
    suggested_render_mask = 3
    if material.opacity < 1:
        suggested_render_mask = 2
    elif shader["color_mode"] == "uniform":
        suggested_render_mask = 3 if material.color[3] >= 1 else 2
    else:
        suggested_render_mask = 3

    # Done
    return [
        {
            "suggested_render_mask": suggested_render_mask,
            "render_shader": shader,
            "primitive_topology": wgpu.PrimitiveTopology.triangle_strip,
            "indices": (n, 1),
            "bindings0": bindings,
            "target": None,  # default
        },
    ]


class LineShader(WorldObjectShader):
    def get_code(self):
        return (
            self.get_definitions()
            + self.more_definitions()
            + self.common_functions()
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

        struct VertexFuncOutput {
            i: i32;
            pos: vec4<f32>;
            thickness_p: f32;
            vec_from_node_p: vec2<f32>;
        };

        struct FragmentOutput {
            [[location(0)]] color: vec4<f32>;
            [[location(1)]] pick: vec4<i32>;
            [[builtin(frag_depth)]] depth : f32;
        };
        """

    def helpers(self):
        return """

        fn get_point_ndc(index:i32) -> vec4<f32> {
            let raw_pos = load_s_positions(index);
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
        fn vs_main(in: VertexInput) -> Varyings {

            let index = i32(in.index);
            let screen_factor = u_stdinfo.logical_size.xy / 2.0;
            let half_thickness:f32 = u_material.thickness * 0.5;  // in logical pixels
            let l2p:f32 = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;

            let result: VertexFuncOutput = get_vertex_result(index, screen_factor, half_thickness, l2p);
            let i0 = result.i;

            var varyings: Varyings;
            varyings.position = vec4<f32>(result.pos);
            varyings.world_pos = vec3<f32>(ndc_to_world_pos(result.pos));
            varyings.thickness_p = f32(result.thickness_p);
            varyings.vec_from_node_p = vec2<f32>(result.vec_from_node_p);

            // Picking
            // Note: in theory, we can store ints up to 16_777_216 in f32,
            // but in practice, its about 4_000_000 for f32 varyings (in my tests).
            // We use a real u32 to not lose presision, see frag shader for details.
            varyings.pick_idx = u32(result.i);
            varyings.pick_zigzag = f32(select(0.0, 1.0, result.i % 2 == 0));

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
        )

    def vertex_shader_line(self):
        return """
        fn get_vertex_result(
            index:i32, screen_factor:vec2<f32>, half_thickness:f32, l2p:f32
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
            // Note that if we sample out of bounds, this affects the shader in mysterious ways (21-12-2021).
            let npos1 = get_point_ndc(max(0, i - 1));
            let npos2 = get_point_ndc(i);
            let npos3 = get_point_ndc(min(u_renderer.last_i, i + 1));

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
            let the_vec = vectors[index % 5] * half_thickness;

            var out : VertexFuncOutput;
            out.i = i;
            out.pos = vec4<f32>((ppos2 + the_vec) / screen_factor - 1.0, npos2.zw);
            out.thickness_p = half_thickness * 2.0 * l2p;
            out.vec_from_node_p = the_vec * l2p;
            return out;
        }
        """

    def vertex_shader_segment(self):
        return """
        fn get_vertex_result(
            index:i32, screen_factor:vec2<f32>, half_thickness:f32, l2p:f32
        ) -> VertexFuncOutput {
            // Similar to the regular line shader, except we only draw segments,
            // using 5 vertices per node. Four for the segments, and 1 to create
            // a degenerate triangle for the space in between. So we only draw
            // caps, no joins.

            let i = index / 5;

            // Sample the current node and either of its neighbours
            let i3 = i + 1 - (i % 2) * 2;  // (i + 1) if i is even else (i - 1)
            let npos2 = get_point_ndc(i);
            let npos3 = get_point_ndc(i3);
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
            let the_vec = vectors[index % 10] * half_thickness;

            var out : VertexFuncOutput;
            out.i = i;
            out.pos = vec4<f32>((ppos2 + the_vec) / screen_factor - 1.0, npos2.zw);
            out.thickness_p = half_thickness * 2.0 * l2p;
            out.vec_from_node_p = the_vec * l2p;
            return out;
        }
        """

    def vertex_shader_arrow(self):
        return """
        fn get_vertex_result(
            index:i32, screen_factor:vec2<f32>, half_thickness:f32, l2p:f32
        ) -> VertexFuncOutput {
            // Similar to the normal vertex shader, except we only draw segments,
            // using 3 vertices per node: 6 per segment. 4 for the arrow, and 2
            // to create a degenerate triangle for the space in between. So we
            // only draw caps, no joins.

            let i = index / 3;

            // Sample the current node and either of its neighbours
            let i3 = i + 1 - (i % 2) * 2;  // (i + 1) if i is even else (i - 1)
            let npos2 = get_point_ndc(i);
            let npos3 = get_point_ndc(i3);
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
                na = normalize(vec2<f32>(v.y, -v.x)) * half_thickness;
                nb = v;
            } else {
                // A right cap
                let v = ppos2.xy - ppos3.xy;
                na = -0.75 * v;
                nb = normalize(vec2<f32>(-v.y, v.x)) * half_thickness - v;
            }

            // Select the correct vector
            // Note the replicated vertices to create degenerate triangles
            var vectors = array<vec2<f32>,6>(na, na, nb, na, nb, nb);
            let the_vec = vectors[index % 6];

            var out : VertexFuncOutput;
            out.i = i;
            out.pos = vec4<f32>((ppos2 + the_vec) / screen_factor - 1.0, npos2.zw);
            out.thickness_p = half_thickness * 2.0 * l2p;
            out.vec_from_node_p = vec2<f32>(0.0, 0.0);
            return out;
        }
        """

    def fragment_shader(self):
        return """
        [[stage(fragment)]]
        fn fs_main(varyings: Varyings) -> FragmentOutput {

            // Discard fragments outside of the radius. This is what makes round
            // joins and caps. If we ever want bevel or miter joins, we should
            // change the vertex positions a bit, and drop these lines below.
            let dist_to_node_p = length(varyings.vec_from_node_p);
            if (dist_to_node_p > varyings.thickness_p * 0.5) {
                discard;
            }

            // Prep
            var alpha: f32 = 1.0;

            // Anti-aliasing. Note that because of the discarding above, we cannot use MSAA.
            // By default, the renderer uses SSAA (super-sampling), but if we apply AA for the edges
            // here this will help the end result. Because this produces semitransparent fragments,
            // it relies on a good blend method, and the object gets drawn twice.
            // todo: because of this aa edge, our line gets a wee bit thinner, so we have to
            // output ticker lines in the vertex shader!
            let aa_width = 1.0;
            alpha = ((0.5 * varyings.thickness_p) - abs(dist_to_node_p)) / aa_width;
            alpha = clamp(alpha, 0.0, 1.0);
            alpha = pow(alpha, 2.0);

            // Set color
            $$ if color_mode == 'vertex'
                let color = varyings.color;
            $$ elif color_mode == 'map'
                let color = sample_colormap(varyings.texcoord);
            $$ else
                let color = u_material.color;
            $$ endif

            let final_color = vec4<f32>(color.rgb, min(1.0, color.a) * alpha * u_material.opacity);

            // Wrap up
            apply_clipping_planes(varyings.world_pos);
            var out = get_fragment_output(varyings.position.z, final_color);

            // Set picking info.
            $$ if write_pick
            // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
            // The pick_idx is int-truncated, so going from a to b, it still has the value of a
            // even right up to b. The pick_zigzag alternates between 0 (even indices) and 1 (odd indices).
            // Here we decode that. The result is that we can support vertex indices of ~32 bits if we want.
            let is_even = varyings.pick_idx % 2u == 0u;
            var coord = select(varyings.pick_zigzag, 1.0 - varyings.pick_zigzag, is_even);
            coord = select(coord, coord - 1.0, coord > 0.5);
            let idx = varyings.pick_idx + select(0u, 1u, coord < 0.0);
            out.pick = (
                pick_pack(u32(u_wobject.id), 20) +
                pick_pack(u32(idx), 26) +
                pick_pack(u32(coord * 100000.0 + 100000.0), 18)
            );
            $$ endif

            // The outer edges with lower alpha for aa are pushed a bit back to avoid artifacts.
            // This is only necessary for blend method "ordered1"
            //out.depth = varyings.position.z + 0.0001 * (0.8 - min(0.8, alpha));

            return out;
        }
        """


@register_wgpu_render_function(Line, LineThinSegmentMaterial)
@register_wgpu_render_function(Line, LineThinMaterial)
def thin_line_renderer(render_info):
    """Render function capable of rendering lines."""

    wobject = render_info.wobject
    material = wobject.material
    geometry = wobject.geometry
    shader = ThinLineShader(
        render_info,
        vertex_color_channels=0,
    )

    positions1 = geometry.positions

    primitive = wgpu.PrimitiveTopology.line_strip
    if isinstance(material, LineThinSegmentMaterial):
        primitive = wgpu.PrimitiveTopology.line_list

    bindings = [
        Binding("u_stdinfo", "buffer/uniform", render_info.stdinfo_uniform),
        Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
        Binding("u_material", "buffer/uniform", material.uniform_buffer),
    ]

    vertex_buffers = {0: positions1}

    # Per-vertex color, colormap, or a plane color?
    shader["color_mode"] = "uniform"
    if material.vertex_colors:
        shader["color_mode"] = "vertex"
        shader["vertex_color_channels"] = nchannels = geometry.colors.data.shape[1]
        if nchannels not in (1, 2, 3, 4):
            raise ValueError(f"Geometry.colors needs 1-4 columns, not {nchannels}")
        vertex_buffers[1] = geometry.colors
    elif material.map is not None:
        shader["color_mode"] = "map"
        bindings.extend(handle_colormap(geometry, material, shader))
        # This shader uses vertex buffers
        bindings.pop(-1)
        vertex_buffers[1] = geometry.texcoords

    # Determine in what render passes this objects must be rendered
    # Note: the renderer use alpha for aa, so we are never opaque only.
    suggested_render_mask = 3
    if material.opacity < 1:
        suggested_render_mask = 2
    elif shader["color_mode"] == "uniform":
        suggested_render_mask = 3 if material.color[3] >= 1 else 2
    else:
        suggested_render_mask = 3

    # Let the shader generate code for our bindings
    for i, binding in enumerate(bindings):
        shader.define_binding(0, i, binding)

    return [
        {
            "suggested_render_mask": suggested_render_mask,
            "render_shader": shader,
            "primitive_topology": primitive,
            "indices": (positions1.nitems, 1),
            "vertex_buffers": vertex_buffers,
            "bindings0": bindings,
        },
    ]


class ThinLineShader(WorldObjectShader):
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
            [[location(0)]] pos : vec3<f32>;
            $$ if color_mode == 'vertex'
                $$ if vertex_color_channels == 1
                [[location(1)]] color : f32;
                $$ else
                [[location(1)]] color : vec{{ vertex_color_channels }}<f32>;
                $$ endif
            $$ elif color_mode == 'map'
                $$ if colormap_dim == '1d'
                [[location(1)]] texcoord : f32;
                $$ elif colormap_dim == '2d'
                [[location(1)]] texcoord : vec2<f32>;
                $$ else
                [[location(1)]] texcoord : vec3<f32>;
                $$ endif
            $$ endif
            [[builtin(vertex_index)]] index : u32;
        };

        [[stage(vertex)]]
        fn vs_main(in: VertexInput) -> Varyings {

            let wpos = u_wobject.world_transform * vec4<f32>(in.pos.xyz, 1.0);
            let npos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos;

            var varyings: Varyings;
            varyings.position = vec4<f32>(npos);
            varyings.world_pos = vec3<f32>(ndc_to_world_pos(npos));

            // Per-vertex colors
            $$ if vertex_color_channels == 1
            varyings.color = vec4<f32>(in.color, in.color, in.color, 1.0);
            $$ elif vertex_color_channels == 2
            varyings.color = vec4<f32>(in.color.r, in.color.r, in.color.r, in.color.g);
            $$ elif vertex_color_channels == 3
            varyings.color = vec4<f32>(in.color, 1.0);
            $$ elif vertex_color_channels == 4
            varyings.color = vec4<f32>(in.color);
            $$ endif

            // Set texture coords
            $$ if colormap_dim == '1d'
            varyings.texcoord = f32(in.texcoord);
            $$ elif colormap_dim == '2d'
            varyings.texcoord = vec2<f32>(in.texcoord);
            $$ elif colormap_dim == '3d'
            varyings.texcoord = vec3<f32>(in.texcoord);
            $$ endif

            return varyings;
        }
        """

    def fragment_shader(self):
        return """
        [[stage(fragment)]]
        fn fs_main(varyings: Varyings) -> FragmentOutput {

            $$ if color_mode == 'vertex'
                let color = varyings.color;
            $$ elif color_mode == 'map'
                let color = sample_colormap(varyings.texcoord);
            $$ else
                let color = u_material.color;
            $$ endif

            let final_color = vec4<f32>(color.rgb, color.a * u_material.opacity);

            apply_clipping_planes(varyings.world_pos);
            return get_fragment_output(varyings.position.z, final_color);
        }
        """
