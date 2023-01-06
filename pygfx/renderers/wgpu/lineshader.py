import wgpu  # only for flags/enums

from . import register_wgpu_render_function, WorldObjectShader, Binding, RenderMask
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
class LineShader(WorldObjectShader):

    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)
        self["line_type"] = "line"
        self["aa"] = wobject.material.aa

    def get_bindings(self, wobject, shared):
        material = wobject.material
        geometry = wobject.geometry

        self["vertex_color_channels"] = 0

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
        if positions1.data.shape[1] != 3:
            raise ValueError(
                "For rendering (thick) lines, the geometry.positions must be Nx3."
            )

        uniform_buffer = Buffer(array_from_shadertype(renderer_uniform_type))
        uniform_buffer.data["last_i"] = positions1.nitems - 1

        rbuffer = "buffer/read_only_storage"
        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
            Binding("u_renderer", "buffer/uniform", uniform_buffer),
            Binding("s_positions", rbuffer, positions1, "VERTEX"),
        ]

        # Per-vertex color, colormap, or a plane color?
        self["color_mode"] = "uniform"
        if material.vertex_colors:
            self["color_mode"] = "vertex"
            self["vertex_color_channels"] = nchannels = geometry.colors.data.shape[1]
            if nchannels not in (1, 2, 3, 4):
                raise ValueError(f"Geometry.colors needs 1-4 columns, not {nchannels}")
            bindings.append(Binding("s_colors", rbuffer, geometry.colors, "VERTEX"))
        elif material.map is not None:
            self["color_mode"] = "map"
            bindings.extend(
                self.define_vertex_colormap(material.map, geometry.texcoords)
            )

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

    def _get_n(self, positions):
        return positions.nitems * 5

    def get_render_info(self, wobject, shared):
        material = wobject.material
        # Determine how many vertices are needed
        n = self._get_n(wobject.geometry.positions)
        # Determine in what render passes this objects must be rendered
        render_mask = wobject.render_mask
        if not render_mask:
            render_mask = RenderMask.all
            if material.is_transparent:
                render_mask = RenderMask.transparent
            elif self["color_mode"] == "uniform":
                if material.color_is_transparent:
                    render_mask = RenderMask.transparent
                elif material.aa:
                    render_mask = RenderMask.all
                else:
                    render_mask = RenderMask.opaque
            elif self["color_mode"] == "vertex":
                if self["vertex_color_channels"] in (2, 4):
                    render_mask = RenderMask.all
                elif material.aa:
                    render_mask = RenderMask.all
                else:
                    render_mask = RenderMask.opaque
            elif self["color_mode"] == "map":
                if self["colormap_nchannels"] in (2, 4):
                    render_mask = RenderMask.all
                elif material.aa:
                    render_mask = RenderMask.all
                else:
                    render_mask = RenderMask.opaque
        return {
            "indices": (n, 1),
            "render_mask": render_mask,
        }

    def get_code(self):
        return (
            self.code_definitions()
            + self.code_more_definitions()
            + self.code_common()
            + self.code_helpers()
            + self.code_vertex()
            + self.code_fragment()
        )

    def code_more_definitions(self):
        return """
        struct VertexInput {
            @builtin(vertex_index) index : u32,
        };

        struct VertexFuncOutput {
            i: i32,
            pos: vec4<f32>,
            thickness_p: f32,
            vec_from_node_p: vec2<f32>,
        };

        struct FragmentOutput {
            @location(0) color: vec4<f32>,
            @location(1) pick: vec4<i32>,
            @builtin(frag_depth) depth : f32,
        };
        """

    def code_helpers(self):
        return """

        fn get_point_ndc(index:i32) -> vec4<f32> {
            let raw_pos = load_s_positions(index);
            let world_pos = u_wobject.world_transform * vec4<f32>(raw_pos.xyz, 1.0);
            return u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;
        }
        """

    def code_vertex(self):
        core = self.code_vertex_core()

        return (
            core
            + """
        @stage(vertex)
        fn vs_main(in: VertexInput) -> Varyings {

            let index = i32(in.index);
            let screen_factor = u_stdinfo.logical_size.xy / 2.0;
            let l2p:f32 = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;
            let extra_thick = {{ '0.5' if aa else '0.0' }} / l2p;
            let half_thickness:f32 = u_material.thickness * 0.5 + extra_thick;  // logical pixels

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

    def code_vertex_core(self):
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

            // Convert to logical screen coordinates, because that's where the lines work
            let ppos1 = (npos1.xy / npos1.w + 1.0) * screen_factor;
            let ppos2 = (npos2.xy / npos2.w + 1.0) * screen_factor;
            let ppos3 = (npos3.xy / npos3.w + 1.0) * screen_factor;

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
            } else if (i == u_renderer.last_i) {
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
                let angle = -atan2( na.x * nc.y - na.y * nc.x, na.x * nc.x + na.y * nc.y );

                // Determine the direction of ne
                let vec_dir = select(na + nc, nb + nd, angle >= 0.0);

                // From the angle we can also determine how long the ne vector should be.
                // We express it in a vector magnifier, and limit it to a factor 2,
                // since when the angle is ~pi, the intersection is near infinity.
                // For a bevel join we can omit ne (or set vec_mag to 1.0).
                // For a miter join we'd need an extra vertex to smoothly transition
                // from a miter to a bevel when the angle is too small.
                let vec_mag = 1.0 / max(0.25, cos(0.5 * angle));

                ne = normalize(vec_dir) * vec_mag;
            }

            // Select the correct vector, note that all except ne are unit.
            var vectors = array<vec2<f32>,5>(na, nb, ne, nc, nd);
            let the_vec = vectors[index % 5] * half_thickness;
            let the_pos = ppos2 + the_vec;

            var out : VertexFuncOutput;
            out.i = i;
            out.pos = vec4<f32>((the_pos / screen_factor - 1.0) * npos2.w, npos2.zw);
            out.thickness_p = half_thickness * 2.0 * l2p;
            out.vec_from_node_p = the_vec * l2p;
            return out;
        }
        """

    def code_fragment(self):
        return """
        @stage(fragment)
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
            $$ if aa
                let aa_width = 1.0;
                alpha = ((0.5 * varyings.thickness_p) - abs(dist_to_node_p)) / aa_width;
                alpha = clamp(alpha, 0.0, 1.0);
            $$ endif

            // Set color
            $$ if color_mode == 'vertex'
                let color = varyings.color;
            $$ elif color_mode == 'map'
                let color = sample_colormap(varyings.texcoord);
            $$ else
                let color = u_material.color;
            $$ endif

            let physical_color = srgb2physical(color.rgb);
            let opacity = min(1.0, color.a) * alpha * u_material.opacity;
            let out_color = vec4<f32>(physical_color, opacity);

            // Wrap up
            apply_clipping_planes(varyings.world_pos);
            var out = get_fragment_output(varyings.position.z, out_color);

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


@register_wgpu_render_function(Line, LineSegmentMaterial)
class LineSegmentShader(LineShader):
    def __init__(self, wobject):
        super().__init__(wobject)
        self["line_type"] = "segment"

    def _get_n(self, positions):
        return (positions.nitems // 2) * 2 * 5

    def code_vertex_core(self):
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
            let ppos2 = (npos2.xy / npos2.w + 1.0) * screen_factor;
            let ppos3 = (npos3.xy / npos3.w + 1.0) * screen_factor;

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
            let the_pos = ppos2 + the_vec;

            var out : VertexFuncOutput;
            out.i = i;
            out.pos = vec4<f32>((the_pos / screen_factor - 1.0) * npos2.w, npos2.zw);
            out.thickness_p = half_thickness * 2.0 * l2p;
            out.vec_from_node_p = the_vec * l2p;
            return out;
        }
        """


@register_wgpu_render_function(Line, LineArrowMaterial)
class LineArrowShader(LineShader):
    def __init__(self, wobject):
        super().__init__(wobject)
        self["line_type"] = "arrow"

    def _get_n(self, positions):
        return (positions.nitems // 2) * 2 * 4

    def code_vertex_core(self):
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
            let ppos2 = (npos2.xy / npos2.w + 1.0) * screen_factor;
            let ppos3 = (npos3.xy / npos3.w + 1.0) * screen_factor;

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
            let the_pos = ppos2 + the_vec;

            var out : VertexFuncOutput;
            out.i = i;
            out.pos = vec4<f32>((the_pos / screen_factor - 1.0) * npos2.w, npos2.zw);
            out.thickness_p = half_thickness * 2.0 * l2p;
            out.vec_from_node_p = vec2<f32>(0.0, 0.0);
            return out;
        }
        """


@register_wgpu_render_function(Line, LineThinMaterial)
class ThinLineShader(WorldObjectShader):

    type = "render"

    def get_bindings(self, wobject, shared):
        material = wobject.material
        geometry = wobject.geometry

        rbuffer = "buffer/read_only_storage"
        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
            Binding("s_positions", rbuffer, geometry.positions, "VERTEX"),
        ]

        # Per-vertex color, colormap, or a plane color?
        self["vertex_color_channels"] = 0
        self["color_mode"] = "uniform"
        if material.vertex_colors:
            self["color_mode"] = "vertex"
            self["vertex_color_channels"] = nchannels = geometry.colors.data.shape[1]
            if nchannels not in (1, 2, 3, 4):
                raise ValueError(f"Geometry.colors needs 1-4 columns, not {nchannels}")
            bindings.append(Binding("s_colors", rbuffer, geometry.colors, "VERTEX"))
        elif material.map is not None:
            self["color_mode"] = "map"
            bindings.extend(
                self.define_vertex_colormap(material.map, geometry.texcoords)
            )

        bindings = {i: b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)

        return {
            0: bindings,
        }

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.line_strip,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        render_mask = wobject.render_mask
        if not render_mask:
            render_mask = RenderMask.all
            if wobject.material.is_transparent:
                render_mask = RenderMask.transparent
            if self["color_mode"] == "uniform":
                if wobject.material.color_is_transparent:
                    render_mask = RenderMask.transparent
                else:
                    render_mask = RenderMask.opaque
        return {
            "indices": (wobject.geometry.positions.nitems, 1),
            "render_mask": render_mask,
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

            let i0 = i32(in.index);

            let raw_pos = load_s_positions(i0);
            let wpos = u_wobject.world_transform * vec4<f32>(raw_pos.xyz, 1.0);
            let npos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos;

            var varyings: Varyings;
            varyings.position = vec4<f32>(npos);
            varyings.world_pos = vec3<f32>(ndc_to_world_pos(npos));

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

    def code_fragment(self):
        return """
        @stage(fragment)
        fn fs_main(varyings: Varyings) -> FragmentOutput {

            $$ if color_mode == 'vertex'
                let color = varyings.color;
            $$ elif color_mode == 'map'
                let color = sample_colormap(varyings.texcoord);
            $$ else
                let color = u_material.color;
            $$ endif

            let physical_color = srgb2physical(color.rgb);
            let opacity = color.a * u_material.opacity;
            let out_color = vec4<f32>(physical_color, opacity);

            apply_clipping_planes(varyings.world_pos);
            return get_fragment_output(varyings.position.z, out_color);
        }
        """


@register_wgpu_render_function(Line, LineThinSegmentMaterial)
class ThinLineSegmentShader(ThinLineShader):
    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.line_list,
            "cull_mode": wgpu.CullMode.none,
        }
