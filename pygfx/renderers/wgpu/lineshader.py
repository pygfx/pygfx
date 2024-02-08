import wgpu  # only for flags/enums
import numpy as np
import pylinalg as la

from . import (
    register_wgpu_render_function,
    WorldObjectShader,
    Binding,
    RenderMask,
    load_shader,
)
from ...utils import array_from_shadertype
from ...resources import Buffer
from ...objects import Line
from ...materials._line import (
    LineMaterial,
    LineDebugMaterial,
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
        self["debug"] = False
        self["aa"] = material.aa
        self["dashing"] = False
        self["thickness_space"] = material.thickness_space

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
            "cull_mode": wgpu.CullMode.none,
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
            node_index: i32,
            face_index: i32,
            pos: vec4<f32>,
            // Varying specifying the thickness of the line, in physical pixels.
            thickness_pw: f32,
            // Varying vector representing the distance from the segment's centerline, in physical pixels.
            segment_coord_pw: vec2<f32>,
            // Varying that is -1 or 1 for the outer corner in a join, for vertex 3 and 4, respectively. Is also used to identify faces that are a join.
            join_coord: f32,
            // Varying that is 1 for vertices in the outer corner of a join. Used in combination with join_coord to obtain a coord that fans around the corner.
            is_outer_corner: f32,
            // Varying used to discard faces in broken joins.
            valid_if_nonzero: f32,
            // Varyings required for dashing
            $$ if dashing
                cumdist_node_w: f32,
                cumdist_vertex_w: f32,
            $$ endif
        };
        """

    def code_helpers(self):
        return """

        fn get_point_world(index:i32) -> vec4<f32> {
            let raw_pos = load_s_positions(index);
            let world_pos = u_wobject.world_transform * vec4<f32>(raw_pos.xyz, 1.0);
            return world_pos;
        }

        // todo: remove?
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

        fn rotate_vec2(v:vec2<f32>, angle:f32) -> vec2<f32> {
            return vec2<f32>(cos(angle) * v.x - sin(angle) * v.y, sin(angle) * v.x + cos(angle) * v.y);
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

            let result: VertexFuncOutput = get_vertex_result(index, screen_factor, l2p);
            let i0 = result.node_index;
            let face_index = result.face_index;

            var varyings: Varyings;
            varyings.position = vec4<f32>(result.pos);
            varyings.world_pos = vec3<f32>(ndc_to_world_pos(result.pos));
            varyings.w = f32(result.pos.w);
            varyings.thickness_pw = f32(result.thickness_pw);
            varyings.segment_coord_pw = vec2<f32>(result.segment_coord_pw);
            varyings.join_coord = f32(result.join_coord);
            varyings.is_outer_corner = f32(result.is_outer_corner);
            varyings.valid_if_nonzero = f32(result.valid_if_nonzero);
            $$ if debug
                let vertex_index = index % 6;
                varyings.bary = vec3<f32>(f32(vertex_index % 3 == 0), f32(vertex_index % 3 == 1), f32(vertex_index % 3 == 2));
            $$ endif
            $$ if dashing
                varyings.cumdist_node_w = f32(result.cumdist_node_w);
                varyings.cumdist_vertex_w = f32(result.cumdist_vertex_w);
            $$ endif

            // Picking
            // Note: in theory, we can store ints up to 16_777_216 in f32,
            // but in practice, its about 4_000_000 for f32 varyings (in my tests).
            // We use a real u32 to not lose presision, see frag shader for details.
            varyings.pick_idx = u32(result.node_index);
            varyings.pick_zigzag = f32(select(0.0, 1.0, result.node_index % 2 == 0));

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
        return load_shader("line_vert_core.wgsl")

    def code_fragment(self):
        return load_shader("line_frag.wgsl")


@register_wgpu_render_function(Line, LineDebugMaterial)
class LineDebugShader(LineShader):
    def __init__(self, wobject):
        super().__init__(wobject)

        self["debug"] = True


@register_wgpu_render_function(Line, LineDashedMaterial)
class LineDashedShader(LineShader):
    needs_bake_function = True

    def __init__(self, wobject):
        super().__init__(wobject)

        self["dashing"] = bool(wobject.material.dash_pattern)
        self["dash_pattern"] = tuple(wobject.material.dash_pattern)
        self["dash_count"] = len(wobject.material.dash_pattern) // 2

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
        if wobject.material.thickness_space == "model":
            # Skip this step if the position data has not changed
            positions_hash = (id(positions_buffer), positions_buffer.rev, dash_offset)
            if positions_hash == self._positions_hash:
                return
            self._positions_hash = positions_hash
            vertex_array = positions_array
        elif wobject.material.thickness_space == "world":
            vertex_array = la.vec_transform(positions_array, wobject.world.matrix)
        else:  # wobject.material.thickness_space == "screen":
            xyz = la.vec_transform(
                positions_array, camera.camera_matrix @ wobject.world.matrix
            )
            vertex_array = xyz[:, :2] * (0.5 * np.array(logical_size))

        # Calculate distances
        distances = np.linalg.norm(vertex_array[1:] - vertex_array[:-1], axis=1)
        distances[~np.isfinite(distances)] = 0.0

        # Store cumulatives
        distance_array[0] = dash_offset
        distances[0] += dash_offset
        np.cumsum(distances, out=distance_array[1:])

        # Mark that the data has changed
        self.line_distance_buffer.update_range(r_offset, r_size)


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

            let node_index = index / 5;
            let face_index = node_index / 2;

            // Sample the current node and either of its neighbours
            let i3 = node_index + 1 - (node_index % 2) * 2;  // (node_index + 1) if node_index is even else (node_index - 1)
            let node2n = get_point_ndc(node_index);
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
            if ((node_index % 2) == 0) {
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
            out.node_index = node_index;
            out.face_index = face_index;
            out.pos = vec4<f32>((the_pos / screen_factor - 1.0) * node2n.w, node2n.zw);
            out.half_thickness_p = half_thickness * l2p;
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

            let node_index = index / 3;
            let face_index = node_index / 2;

            // Sample the current node and either of its neighbours
            let i3 = node_index + 1 - (node_index % 2) * 2;  // (node_index + 1) if node_index is even else (node_index - 1)
            let node2n = get_point_ndc(node_index);
            let node3n = get_point_ndc(i3);
            // Convert to logical screen coordinates, because that's were the lines work
            let node2s = (node2n.xy / node2n.w + 1.0) * screen_factor;
            let node3s = (node3n.xy / node3n.w + 1.0) * screen_factor;

            // Get vectors normal to the line segments
            var vert1: vec2<f32>;
            var vert2: vec2<f32>;

            // Get vectors normal to the line segments
            if ((node_index % 2) == 0) {
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
            out.node_index = node_index;
            out.face_index = face_index;
            out.pos = vec4<f32>((the_pos / screen_factor - 1.0) * node2n.w, node2n.zw);
            out.half_thickness_p = half_thickness * l2p;
            out.segment_coord_p = vec2<f32>(0.0, 0.0);
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
