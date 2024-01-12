import wgpu  # only for flags/enums
import numpy as np
import pylinalg as la

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
    LineDashedMaterial,
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

# todo: --> http://jcgt.org/published/0002/02/08/paper.pdf


renderer_uniform_type = dict(last_i="i4")


@register_wgpu_render_function(Line, LineMaterial)
class LineShader(WorldObjectShader):
    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)
        material = wobject.material
        geometry = wobject.geometry

        self["line_type"] = "line"
        self["aa"] = material.aa

        color_mode = str(material.color_mode).split(".")[-1]
        if color_mode == "auto":
            if material.map is not None:
                self["color_mode"] = "vertex_map"
                self["color_buffer_channels"] = 0
            else:
                self["color_mode"] = "uniform"
                self["color_buffer_channels"] = 0
        elif color_mode == "uniform":
            self["color_mode"] = "uniform"
            self["color_buffer_channels"] = 0
        elif color_mode == "vertex":
            self["color_mode"] = "vertex"
            self["color_buffer_channels"] = nchannels = geometry.colors.data.shape[1]
            if nchannels not in (1, 2, 3, 4):
                raise ValueError(f"Geometry.colors needs 1-4 columns, not {nchannels}")
        elif color_mode == "face":
            self["color_mode"] = "face"
            self["color_buffer_channels"] = nchannels = geometry.colors.data.shape[1]
            if nchannels not in (1, 2, 3, 4):
                raise ValueError(f"Geometry.colors needs 1-4 columns, not {nchannels}")
        elif color_mode == "vertex_map":
            if material.map is None:
                raise ValueError(f"Cannot apply colormap is no material.map is set.")
            self["color_mode"] = "vertex_map"
            self["color_buffer_channels"] = 0
        elif color_mode == "face_map":
            if material.map is None:
                raise ValueError(f"Cannot apply colormap is no material.map is set.")
            self["color_mode"] = "face_map"
            self["color_buffer_channels"] = 0
        else:
            raise RuntimeError(f"Unknown color_mode: '{color_mode}'")

    def get_bindings(self, wobject, shared):
        material = wobject.material
        geometry = wobject.geometry

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

        # Per-vertex color, colormap, or a uniform color?
        if self["color_mode"] in ("vertex", "face"):
            bindings.append(Binding("s_colors", rbuffer, geometry.colors, "VERTEX"))
        elif self["color_mode"] in ("vertex_map", "face_map"):
            bindings.extend(
                self.define_texcoords_and_colormap(
                    material.map, geometry.texcoords, material.map_interpolation
                )
            )

        bindings.extend(self._get_extra_bindings(wobject))

        bindings = {i: b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)

        return {
            0: bindings,
        }

    def _get_extra_bindings(self, wobject):
        return []  # for subclasses to provide more bindings

    def get_pipeline_info(self, wobject, shared):
        # Cull backfaces so that overlapping faces are not drawn.
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_strip,
            "cull_mode": wgpu.CullMode.front,  # todo: flip something in shader sp we can cull the back
        }

    def _get_n(self, positions):
        offset, size = positions.draw_range
        return offset * 6, size * 6

    def get_render_info(self, wobject, shared):
        material = wobject.material
        # Determine how many vertices are needed
        offset, size = self._get_n(wobject.geometry.positions)
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
            elif self["color_mode"] in ("vertex", "face"):
                if self["color_buffer_channels"] in (2, 4):
                    render_mask = RenderMask.all
                elif material.aa:
                    render_mask = RenderMask.all
                else:
                    render_mask = RenderMask.opaque
            elif self["color_mode"] in ("vertex_map", "face_map"):
                if self["colormap_nchannels"] in (2, 4):
                    render_mask = RenderMask.all
                elif material.aa:
                    render_mask = RenderMask.all
                else:
                    render_mask = RenderMask.opaque
            else:
                raise RuntimeError(f"Unexpected color mode {self['color_mode']}")

        return {
            "indices": (size, 1, offset, 0),
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
            fi: i32,
            pos: vec4<f32>,
            thickness_p: f32,
            vec_from_line_p: vec2<f32>,
            vec_from_node_p: vec2<f32>,
            is_join: f32,
            side: f32,
            cum_dist: f32,
        };
        """

    def code_helpers(self):
        return """

        fn get_point_ndc(index:i32) -> vec4<f32> {
            let raw_pos = load_s_positions(index);
            let world_pos = u_wobject.world_transform * vec4<f32>(raw_pos.xyz, 1.0);
            return u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;
        }

        fn is_nan_or_zero(v:f32) -> bool {
            // Naga has removed isNan checks, because backends may be using fast-math,
            // in which case nan is assumed not to happen, and isNan would always be false.
            // If we assume that some nan mechanics still work, we can still detect it.
            // This won't work however: `return v != v`, because the compiler will
            // optimize it out. The same holds for similar constructs.
            // Maybe the same happens if we turn `<`  into `<=`.
            // So we and up with an equation that detects either NaN or zero,
            // which is fine if we use it on a .w attribute.
            return !(v < 0.0) && !(v > 0.0);
        }
        """

    def code_vertex(self):
        core = self.code_vertex_core()

        return (
            core
            + """
        @vertex
        fn vs_main(in: VertexInput) -> Varyings {

            let index = i32(in.index);
            let screen_factor = u_stdinfo.logical_size.xy / 2.0;
            let l2p:f32 = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;
            let extra_thick = {{ '0.5' if aa else '0.0' }} / l2p;
            let half_thickness:f32 = u_material.thickness * 0.5 + extra_thick;  // logical pixels

            let result: VertexFuncOutput = get_vertex_result(index, screen_factor, half_thickness, l2p);
            let i0 = result.i;
            let face_index = result.fi;

            var varyings: Varyings;
            varyings.position = vec4<f32>(result.pos);
            varyings.world_pos = vec3<f32>(ndc_to_world_pos(result.pos));
            varyings.thickness_p = f32(result.thickness_p);
            varyings.vec_from_line_p = vec2<f32>(result.vec_from_line_p);
            varyings.vec_from_node_p = vec2<f32>(result.vec_from_node_p);
            varyings.is_join = f32(result.is_join);
            varyings.side = f32(result.side);

            //varyings.cum_dist = f32(result.cum_dist);

            // TODO: remove result.cum_dist
            // TODO: fix this hack, bc it breaks non-dashed lines :P
            varyings.cum_dist = f32(load_s_cumdist(i0));

            // Picking
            // Note: in theory, we can store ints up to 16_777_216 in f32,
            // but in practice, its about 4_000_000 for f32 varyings (in my tests).
            // We use a real u32 to not lose presision, see frag shader for details.
            varyings.pick_idx = u32(result.i);
            varyings.pick_zigzag = f32(select(0.0, 1.0, result.i % 2 == 0));

            // per-vertex or per-face coloring
            $$ if color_mode == 'face' or color_mode == 'vertex'
                $$ if color_mode == 'face'
                let color_index = face_index;
                $$ else
                    let color_index = i0;
                $$ endif
                $$ if color_buffer_channels == 1
                    let cvalue = load_s_colors(color_index);
                    varyings.color = vec4<f32>(cvalue, cvalue, cvalue, 1.0);
                $$ elif color_buffer_channels == 2
                    let cvalue = load_s_colors(color_index);
                    varyings.color = vec4<f32>(cvalue.r, cvalue.r, cvalue.r, cvalue.g);
                $$ elif color_buffer_channels == 3
                    varyings.color = vec4<f32>(load_s_colors(color_index), 1.0);
                $$ elif color_buffer_channels == 4
                    varyings.color = vec4<f32>(load_s_colors(color_index));
                $$ endif
            $$ endif

            // How to index into tex-coords
            $$ if color_mode == 'face_map'
            let tex_coord_index = face_index;
            $$ else
            let tex_coord_index = i0;
            $$ endif

            // Set texture coords
            $$ if colormap_dim == '1d'
            varyings.texcoord = f32(load_s_texcoords(tex_coord_index));
            $$ elif colormap_dim == '2d'
            varyings.texcoord = vec2<f32>(load_s_texcoords(tex_coord_index));
            $$ elif colormap_dim == '3d'
            varyings.texcoord = vec3<f32>(load_s_texcoords(tex_coord_index));
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
            //
            // This vertex shader uses VertexId and storage buffers instead of
            // vertex buffers. It creates 6 vertices for each point on the line.
            // The extra vertices are used to cover more fragments at
            // the joins and caps. In the fragment shader we discard fragments
            // that are "out of range" for the current join/cap shape, using
            // parameters passed as varyings.
            //
            // Definitions:
            //
            // - node: the positions that define the line. In other contexts these
            //   may be called vertices or points.
            // - vertex: the "virtual vertices" generated in the vertex shader,
            //   in order to create a thick line with nice joins and caps.
            // - segment: the straight piece of the line between two consecutive
            //   nodes. A quadrilateral (two faces) but not necesarily rectangular.
            // - join: the piece of the line to connect two segments. There are
            //   a few different shapes that can be applied.
            // - cap: the beginning/end of the line and dashes. It typically extends
            //   a bit beyond the node (or dash end). There are multiple cap shapes.
            // - dash: the visible contiguous piece of the line when dashing is
            //   enabled. Can go over a join, i.e. is not always straight. Has caps.
            // - broken join: joins with too sharp corners are rendered as two
            //   separate segments with caps.
            //
            // Basic algorithm and definitions:
            //
            // - We read the positions of three nodes, the current, previous, and next.
            // - These are converted to logical pixel screen space.
            // - We define six normal vectors which represent the (virtual) vertices.
            //   The first two close the previous segment, the last two start the next
            //   segment, the two in the middle help define the join.
            // - These calculations are done for each vertex (yeah, bit of a waste),
            //   we select just one as output.
            //
            //            /  o     node 3
            //           /  /  /
            //          6  /  /
            //   - - - 2  /  /     segment-vertices 1, 2, 5, 6
            //   o-------o  /      the vertices 3 and 4 are in between to help the join
            //   - - - - 1 5
            //                node 2
            //  node 1
            //
            //
            // Possible improvements:
            //
            // - we can prepare the nodes' screen coordinates in a compute shader.

            // Indexing
            let i = index / 6;
            let sub_index = index % 6;
            let fi = (index + 2) / 6;

            // Sample the current node and it's two neighbours, and convert to NDC
            // Note that if we sample out of bounds, this affects the shader in mysterious ways (21-12-2021).
            let node1n = get_point_ndc(max(0, i - 1));
            let node2n = get_point_ndc(i);
            let node3n = get_point_ndc(min(u_renderer.last_i, i + 1));

            // Convert to logical screen coordinates, because that's where the lines work
            let node1s = (node1n.xy / node1n.w + 1.0) * screen_factor;
            let node2s = (node2n.xy / node2n.w + 1.0) * screen_factor;
            let node3s = (node3n.xy / node3n.w + 1.0) * screen_factor;

            // Get vectors representing the two incident line segments
            var nodevec1: vec2<f32> = node2s.xy - node1s.xy;
            var nodevec2: vec2<f32> = node3s.xy - node2s.xy;

            // Declare (relative) vectors representing the 6 vertices.
            // These are relarive to node2 (in screen space).
            var vert1: vec2<f32>;
            var vert2: vec2<f32>;
            var vert3: vec2<f32>;
            var vert4: vec2<f32>;
            var vert5: vec2<f32>;
            var vert6: vec2<f32>;

            // Declare matching line cords (x along line, y perpendicular to it)
            var coord1: vec2<f32>;
            var coord2: vec2<f32>;
            var coord3: vec2<f32>;
            var coord4: vec2<f32>;
            var coord5: vec2<f32>;
            var coord6: vec2<f32>;

            // Whether the current vertex represents the join. Only nonzero for
            // subindex 2 or 3, the signs is -1 and +1, respectively, signaling the side.
            // In the fragmnent shader this is used to determine whether the
            // vec_from_line or vec_from_node is used as the coord to sample the shape.
            var is_join = 0.0;

            var vectors_ll_corner = array<vec2<f32>,6>(coord3, coord4, coord3, coord4, coord3, coord4);

            if ( i == 0 || is_nan_or_zero(node1n.w) ) {
                // This is the first point on the line: create a cap.
                nodevec1 = nodevec2;

                vert5 = normalize(vec2<f32>(nodevec2.y, -nodevec2.x));
                vert6 = -vert5;
                vert3 = vert5 - normalize(nodevec2);  // location of first vertex
                vert4 = vert6 - normalize(nodevec2);

                // Unused vertices go into first vertex
                vert1 = vert3;
                vert2 = vert3;

                coord1 = vert1;
                coord2 = vert2;
                coord3 = vert3;
                coord4 = vert4;
                coord5 = vert5;
                coord6 = vert6;

            } else if ( i == u_renderer.last_i || is_nan_or_zero(node3n.w) )  {
                // This is the last point on the line: create a cap.
                nodevec2 = nodevec1;

                vert1 = normalize(vec2<f32>(nodevec1.y, -nodevec1.x));
                vert2 = -vert1;
                vert3 = vert1 + normalize(nodevec1);
                vert4 = vert2 + normalize(nodevec1);  // location of last vertex

                 // Unused vertices go into last vertex
                vert5 = vert4;
                vert6 = vert4;

                coord1 = vert1;
                coord2 = vert2;
                coord3 = vert3;
                coord4 = vert4;
                coord5 = vert5;
                coord6 = vert6;

            } else {
                // Create a join

                // Outer vertices are straightforward
                vert1 = normalize(vec2<f32>(nodevec1.y, -nodevec1.x));
                vert2 = -vert1;
                vert5 = normalize(vec2<f32>(nodevec2.y, -nodevec2.x));
                vert6 = -vert5;

                // Determine the angle of the corner. If this angle is smaller than zero,
                // the inside of the join is at vert2/vert6, otherwise it is at vert1/vert5.
                let angle = -atan2( nodevec1.x * nodevec2.y - nodevec1.y * nodevec2.x,
                                    nodevec1.x * nodevec2.x + nodevec1.y * nodevec2.y );

                // Determine the direction of vert3 and vert4
                let inner_corner_is_at_135 = angle >= 0.0;

                // The direction in which to place the vert3 and vert4.
                let join_vec = normalize(vert1 + vert5);

                // Now calculate how far along this vector we can still without
                // introducing overlapping faces, which would result in glitchy artifacts.
                let nodevec1_norm = normalize(nodevec1);
                let nodevec2_norm = normalize(nodevec2);
                let join_vec_on_nodevec1 = dot(join_vec, nodevec1_norm) * nodevec1_norm;
                let join_vec_on_nodevec2 = dot(join_vec, nodevec2_norm) * nodevec2_norm;
                var max_vec_mag = 100.0;
                max_vec_mag = min(max_vec_mag, 0.99 * length(nodevec1) / length(join_vec_on_nodevec1) / half_thickness);
                max_vec_mag = min(max_vec_mag, 0.99 * length(nodevec2) / length(join_vec_on_nodevec2) / half_thickness);

                // Now use the angle to determine the join_vec magnitude required to draw this join.
                // For the inner corner this represent the intersection of the line edges,
                // i.e. the point where we should move the two other vertices-at-the-inner-corner to.
                // For the outer corner this represents the miter,
                // i.e. the extra space we need to draw the join shape.
                // Note that when the angle is ~pi, the magnitude is near infinity.
                let vec_mag = 1.0 / cos(0.5 * angle);

                // Clamp the magnitude with the limit we calculated above.
                let vec_mag_clamped = clamp(vec_mag, 1.0, max_vec_mag);

                // If the magnitude got clamped, we cannot draw the join as a contiguous line.
                var join_is_contiguous = vec_mag_clamped == vec_mag;

                if (false) {
                    // Miter
                    // TODO: do this using templating
                } else if (join_is_contiguous) {
                    // Round or miter, shallow (enough) corner

                    vert3 = join_vec * vec_mag_clamped;
                    vert4 = -vert3;

                    coord1 = vert1;
                    coord2 = vert2;
                    coord3 = vert3;
                    coord4 = vert4;
                    coord5 = vert5;
                    coord6 = vert6;

                    // Put the 3 vertices in the inner corner at the same (center) position.
                    // Adjust the corner_coords in the same way, or they would not be correct.

                    // TODO: rename vectors_ll_corner -> node_coord, being vec to the node.
                    // TODO: move this bit to the root and end of the function?
                    if (inner_corner_is_at_135) {
                        vert1 = vert3;
                        vert5 = vert3;
                        is_join = f32(sub_index == 3);
                        vectors_ll_corner[0] = coord3;
                        vectors_ll_corner[1] = coord2;
                        vectors_ll_corner[2] = coord3;
                        vectors_ll_corner[3] = coord4;
                        vectors_ll_corner[4] = coord3;
                        vectors_ll_corner[5] = coord6;
                    } else {
                        vert2 = vert4;
                        vert6 = vert4;
                        is_join = -f32(sub_index == 2);
                        vectors_ll_corner[0] = coord1;
                        vectors_ll_corner[1] = coord4;
                        vectors_ll_corner[2] = coord3;
                        vectors_ll_corner[3] = coord4;
                        vectors_ll_corner[4] = coord5;
                        vectors_ll_corner[5] = coord4;
                    }

                } else {
                    // Broken join: render as separate segments with caps.

                    // Place the two middle point to form a miter that is long
                    // enough to draw a good-looking round cap. The face between
                    // the miters is flipped and therefore culled.
                    vert3 = normalize(nodevec1) * 4.0;
                    vert4 = normalize(-nodevec2) * 4.0;

                    coord1 = vert1;
                    coord2 = vert2;
                    coord3 = vert3;
                    coord4 = vert4;
                    coord5 = vert5;
                    coord6 = vert6;
                }
            }

            // Select the current vector.
            var vert_array = array<vec2<f32>,6>(vert1, vert2, vert3, vert4, vert5, vert6);
            let the_vert_s = vert_array[index % 6] * half_thickness;
            let the_pos_s = node2s + the_vert_s;
            let the_pos_n = vec4<f32>((the_pos_s / screen_factor - 1.0) * node2n.w, node2n.zw);
            var coord_array = array<vec2<f32>,6>(coord1, coord2, coord3, coord4, coord5, coord6);

            let vec_from_line_p = coord_array[sub_index];
            let the_ll_vec_corner = vectors_ll_corner[sub_index];

            // Calculate side
            let side = (f32(index % 2) * 2.0 - 1.0) * length(the_ll_vec_corner);

            var out : VertexFuncOutput;
            out.i = i;
            out.fi = fi;
            out.pos = the_pos_n;
            out.thickness_p = half_thickness * 2.0 * l2p;
            //out.vec_from_line_p = the_vert_s * 2.0 * l2p; // TODO: rename
            out.vec_from_line_p = vec_from_line_p;
            out.vec_from_node_p = the_ll_vec_corner;
            out.is_join = is_join;
            out.side = side;
            out.cum_dist = 0.0;
            return out;
        }
        """

    def code_fragment(self):
        return """
        @fragment
        fn fs_main(varyings: Varyings, @builtin(front_facing) is_front: bool) -> FragmentOutput {

            // Discard fragments outside of the radius. This is what makes round
            // joins and caps. If we ever want bevel or miter joins, we should
            // change the vertex positions a bit, and drop these lines below.

            // Butt cap
            //if (varyings.vec_from_line_p.x > 0.0) {
            //     discard;
            //}

            let is_join = varyings.is_join != 0.0;
            let line_coord_p = select(varyings.vec_from_line_p, varyings.vec_from_node_p, is_join);

            let free_zone = (varyings.is_join * varyings.side) < 0.0;

            let dist_to_node_p = length(line_coord_p);
            //if (dist_to_node_p > varyings.thickness_p) {
            if (dist_to_node_p > 1.0 && !free_zone) {
               discard;
            }

            // Prep
            var alpha: f32 = 1.0;

            // Anti-aliasing. Note that because of the discarding above, we cannot use MSAA.
            // By default, the renderer uses SSAA (super-sampling), but if we apply AA for the edges
            // here this will help the end result. Because this produces semitransparent fragments,
            // it relies on a good blend method, and the object gets drawn twice.
            $$ if false
                let aa_width = 1.0;
                alpha = ((0.5 * varyings.thickness_p) - abs(dist_to_node_p)) / aa_width;
                alpha = clamp(alpha, 0.0, 1.0);
            $$ endif

            $$ if color_mode == 'vertex' or color_mode == 'face'
                let color = varyings.color;
            $$ elif color_mode == 'vertex_map' or color_mode == 'face_map'
                let color = sample_colormap(varyings.texcoord);
            $$ else
                let color = u_material.color;
            $$ endif

            var physical_color = srgb2physical(color.rgb);
            if (false) {
                physical_color = vec3<f32>(1.0, 0.0, 0.0);
            }
            let opacity = min(1.0, color.a) * alpha * u_material.opacity;

            //let opacity_multiplier = select(-1.0, 1.0, !is_front);
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


@register_wgpu_render_function(Line, LineDashedMaterial)
class LineDashedShader(LineShader):
    needs_bake_function = True

    def __init__(self, wobject):
        super().__init__(wobject)

        n_verts = wobject.geometry.positions.nitems
        distance_array = np.zeros((n_verts,), np.float32)
        self.line_distance_buffer = Buffer(distance_array)
        self._positions_hash = None

    def _get_extra_bindings(self, wobject):
        rbuffer = "buffer/read_only_storage"
        return [
            Binding("s_cumdist", rbuffer, self.line_distance_buffer, "VERTEX"),
        ]

    def bake_function(self, wobject, camera, logical_size):
        # Prepare
        positions_buffer = wobject.geometry.positions
        dash_offset = wobject.material.dash_offset
        r_offset, r_size = positions_buffer.draw_range

        # Prepare arrays
        positions_array = positions_buffer.data[r_offset : r_offset + r_size]
        distance_array = self.line_distance_buffer.data[r_offset : r_offset + r_size]

        # Get vertices in the appropriate coordinate frame
        if wobject.material.dash_is_screen_space:
            xyz = la.vec_transform(positions_array, camera.camera_matrix)
            vertex_array = xyz[:, :2] * 0.5 * np.array(logical_size)
        else:
            # Skip this step if the position data has not changed
            positions_hash = (id(positions_buffer), positions_buffer.rev, dash_offset)
            if positions_hash == self._positions_hash:
                return
            self._positions_hash = positions_hash
            vertex_array = positions_array

        # Calculate distances
        distances = np.linalg.norm(vertex_array[1:] - vertex_array[:-1], axis=1)
        distances[~np.isfinite(distances)] = 0.0

        # Store cumulatives
        distance_array[0] = dash_offset
        distances[0] += dash_offset
        np.cumsum(distances, out=distance_array[1:])

        # Mark that the data has changed
        self.line_distance_buffer.update_range(r_offset, r_size)

    def code_fragment(self):
        return """
        @fragment
        fn fs_main(varyings: Varyings) -> FragmentOutput {

            let line_coord_p = varyings.vec_from_line_p;

            let dash_size = u_material.dash_size;
            let dash_ratio = u_material.dash_ratio;
            let dash_progress = (varyings.cum_dist % dash_size) / dash_size;

            // Get distance to dash-stroke. We make the stroke the center
            // of the dash period, which makes the math easier.
            //
            //        ratio e.g. 0.6
            //       /       \
            //  ----|---------|----
            //  0       0.5       1    dash_progress
            // 0.2  0  -0.3   0  0.2   dist_to_stroke

            // TODO: perhaps an offset to cum_dist to make it start with a stroke?
            var dist_to_stroke = abs(dash_progress - 0.5) - 0.5 * dash_ratio;
            dist_to_stroke = max(0.0, dist_to_stroke);

            // Convert to pixel units
            let dpd_cumdist = length(vec2<f32>(dpdxFine(varyings.cum_dist), dpdyFine(varyings.cum_dist)));
            let dist_to_stroke_p = dash_size * dist_to_stroke/dpd_cumdist;

            // Butt caps
            if (dist_to_stroke > 0.0) {
            //    discard;
            }

            // Round caps
            let vec_from_gap = vec2<f32>(dist_to_stroke_p, length(line_coord_p));
            if (length(vec_from_gap) > varyings.thickness_p * 0.5) {
                discard;
            }

            // Discard fragments outside of the radius. This is what makes round
            // joins and caps. If we ever want bevel or miter joins, we should
            // change the vertex positions a bit, and drop these lines below.
            let dist_to_node_p = length(line_coord_p);
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

            $$ if color_mode == 'vertex' or color_mode == 'face'
                let color = varyings.color;
            $$ elif color_mode == 'vertex_map' or color_mode == 'face_map'
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
        offset, size = positions.draw_range
        return (offset // 2) * 2 * 5, (size // 2) * 2 * 5

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
            let fi = i / 2;

            // Sample the current node and either of its neighbours
            let i3 = i + 1 - (i % 2) * 2;  // (i + 1) if i is even else (i - 1)
            let node2n = get_point_ndc(i);
            let node3n = get_point_ndc(i3);
            // Convert to logical screen coordinates, because that's were the lines work
            let node2s = (node2n.xy / node2n.w + 1.0) * screen_factor;
            let node3s = (node3n.xy / node3n.w + 1.0) * screen_factor;

            // Get vectors normal to the line segments
            var vert1: vec2<f32>;
            var vert2: vec2<f32>;
            var vert3: vec2<f32>;
            var vert4: vec2<f32>;

            // Get vectors normal to the line segments
            if ((i % 2) == 0) {
                // A left-cap
                let v = normalize(node3s.xy - node2s.xy);
                vert3 = vec2<f32>(v.y, -v.x);
                vert4 = -vert3;
                vert1 = vert3 - v;
                vert2 = vert4 - v;
            } else {
                // A right cap
                let v = normalize(node2s.xy - node3s.xy);
                vert1 = vec2<f32>(v.y, -v.x);
                vert2 = -vert1;
                vert3 = vert1 + v;
                vert4 = vert2 + v;
            }

            // Select the correct vector
            // Note the replicated vertices to create degenerate triangles
            var vectors = array<vec2<f32>,10>(vert1, vert1, vert2, vert3, vert4, vert1, vert2, vert3, vert4, vert4);
            let the_vec = vectors[index % 10] * half_thickness;
            let the_pos = node2s + the_vec;

            var out : VertexFuncOutput;
            out.i = i;
            out.fi = fi;
            out.pos = vec4<f32>((the_pos / screen_factor - 1.0) * node2n.w, node2n.zw);
            out.thickness_p = half_thickness * 2.0 * l2p;
            out.vec_from_line_p = the_vec * l2p;
            return out;
        }
        """


@register_wgpu_render_function(Line, LineArrowMaterial)
class LineArrowShader(LineShader):
    def __init__(self, wobject):
        super().__init__(wobject)
        self["line_type"] = "arrow"

    def _get_n(self, positions):
        offset, size = positions.draw_range
        return (offset // 2) * 2 * 4, (size // 2) * 2 * 4

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
            let fi = i / 2;

            // Sample the current node and either of its neighbours
            let i3 = i + 1 - (i % 2) * 2;  // (i + 1) if i is even else (i - 1)
            let node2n = get_point_ndc(i);
            let node3n = get_point_ndc(i3);
            // Convert to logical screen coordinates, because that's were the lines work
            let node2s = (node2n.xy / node2n.w + 1.0) * screen_factor;
            let node3s = (node3n.xy / node3n.w + 1.0) * screen_factor;

            // Get vectors normal to the line segments
            var vert1: vec2<f32>;
            var vert2: vec2<f32>;

            // Get vectors normal to the line segments
            if ((i % 2) == 0) {
                // A left-cap
                let v = node3s.xy - node2s.xy;
                vert1 = normalize(vec2<f32>(v.y, -v.x)) * half_thickness;
                vert2 = v;
            } else {
                // A right cap
                let v = node2s.xy - node3s.xy;
                vert1 = -0.75 * v;
                vert2 = normalize(vec2<f32>(-v.y, v.x)) * half_thickness - v;
            }

            // Select the correct vector
            // Note the replicated vertices to create degenerate triangles
            var vectors = array<vec2<f32>,6>(vert1, vert1, vert2, vert1, vert2, vert2);
            let the_vec = vectors[index % 6];
            let the_pos = node2s + the_vec;

            var out : VertexFuncOutput;
            out.i = i;
            out.fi = fi;
            out.pos = vec4<f32>((the_pos / screen_factor - 1.0) * node2n.w, node2n.zw);
            out.thickness_p = half_thickness * 2.0 * l2p;
            out.vec_from_line_p = vec2<f32>(0.0, 0.0);
            return out;
        }
        """


@register_wgpu_render_function(Line, LineThinMaterial)
class ThinLineShader(LineShader):
    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)
        self["aa"] = False  # no aa with thin lines
        if self["color_mode"] in ("face", "face_map"):
            raise RuntimeError("Face coloring not supported for thin lines.")

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

        # Per-vertex color, colormap, or a uniform color?
        if self["color_mode"] == "vertex":
            bindings.append(Binding("s_colors", rbuffer, geometry.colors, "VERTEX"))
        elif self["color_mode"] == "vertex_map":
            bindings.extend(
                self.define_texcoords_and_colormap(
                    material.map, geometry.texcoords, material.map_interpolation
                )
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
        material = wobject.material
        offset, size = wobject.geometry.positions.draw_range
        render_mask = wobject.render_mask
        if not render_mask:
            render_mask = RenderMask.all
            if material.is_transparent:
                render_mask = RenderMask.transparent
            elif self["color_mode"] == "uniform":
                if material.color_is_transparent:
                    render_mask = RenderMask.transparent
                else:
                    render_mask = RenderMask.opaque
            elif self["color_mode"] == "vertex":
                if self["color_buffer_channels"] in (2, 4):
                    render_mask = RenderMask.all
                else:
                    render_mask = RenderMask.opaque
            elif self["color_mode"] == "vertex_map":
                if self["colormap_nchannels"] in (2, 4):
                    render_mask = RenderMask.all
                else:
                    render_mask = RenderMask.opaque
        return {
            "indices": (size, 1, offset, 0),
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

        @vertex
        fn vs_main(in: VertexInput) -> Varyings {

            let i0 = i32(in.index);

            let raw_pos = load_s_positions(i0);
            let wpos = u_wobject.world_transform * vec4<f32>(raw_pos.xyz, 1.0);
            let npos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos;

            var varyings: Varyings;
            varyings.position = vec4<f32>(npos);
            varyings.world_pos = vec3<f32>(ndc_to_world_pos(npos));

            // per-vertex or per-face coloring
            $$ if color_mode == 'vertex'
                let color_index = i0;
                $$ if color_buffer_channels == 1
                    let cvalue = load_s_colors(color_index);
                    varyings.color = vec4<f32>(cvalue, cvalue, cvalue, 1.0);
                $$ elif color_buffer_channels == 2
                    let cvalue = load_s_colors(color_index);
                    varyings.color = vec4<f32>(cvalue.r, cvalue.r, cvalue.r, cvalue.g);
                $$ elif color_buffer_channels == 3
                    varyings.color = vec4<f32>(load_s_colors(color_index), 1.0);
                $$ elif color_buffer_channels == 4
                    varyings.color = vec4<f32>(load_s_colors(color_index));
                $$ endif
            $$ endif

            // Set texture coords
            let tex_coord_index = i0;
            $$ if colormap_dim == '1d'
            varyings.texcoord = f32(load_s_texcoords(tex_coord_index));
            $$ elif colormap_dim == '2d'
            varyings.texcoord = vec2<f32>(load_s_texcoords(tex_coord_index));
            $$ elif colormap_dim == '3d'
            varyings.texcoord = vec3<f32>(load_s_texcoords(tex_coord_index));
            $$ endif

            return varyings;
        }
        """

    def code_fragment(self):
        return """
        @fragment
        fn fs_main(varyings: Varyings) -> FragmentOutput {

            $$ if color_mode == 'vertex'
                let color = varyings.color;
            $$ elif color_mode == 'vertex_map'
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
