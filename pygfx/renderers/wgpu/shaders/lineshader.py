"""pygfx line shader. See line.wgsl for details."""

import wgpu  # only for flags/enums
import numpy as np
import pylinalg as la

from ....utils import array_from_shadertype
from ....resources import Buffer
from ....objects import Line, InstancedLine
from ....materials._line import (
    LineMaterial,
    LineSegmentMaterial,
    LineInfiniteSegmentMaterial,
    LineArrowMaterial,
    LineThinMaterial,
    LineThinSegmentMaterial,
    LineDebugMaterial,
)

from .. import (
    register_wgpu_render_function,
    BaseShader,
    Binding,
    load_wgsl,
    nchannels_from_format,
)


renderer_uniform_type = dict(last_i="i4")


@register_wgpu_render_function(Line, LineMaterial)
class LineShader(BaseShader):
    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)
        material = wobject.material
        geometry = wobject.geometry

        # Is this an instanced line?
        self["instanced"] = isinstance(wobject, InstancedLine)

        self["line_type"] = "line"
        self["dashing"] = False
        self["thickness_space"] = material.thickness_space
        self["aa"] = material._gfx_effective_aa
        self["loop"] = False
        self["debug"] = False

        # Handle color
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
            nchannels = nchannels_from_format(geometry.colors.format)
            self["color_mode"] = "vertex"
            self["color_buffer_channels"] = nchannels
            if nchannels not in (1, 2, 3, 4):
                raise ValueError(f"Geometry.colors needs 1-4 columns, not {nchannels}")
        elif color_mode == "face":
            nchannels = nchannels_from_format(geometry.colors.format)
            self["color_mode"] = "face"
            self["color_buffer_channels"] = nchannels
            if nchannels not in (1, 2, 3, 4):
                raise ValueError(f"Geometry.colors needs 1-4 columns, not {nchannels}")
        elif color_mode == "vertex_map":
            if material.map is None:
                raise ValueError("Cannot apply colormap is no material.map is set.")
            self["color_mode"] = "vertex_map"
            self["color_buffer_channels"] = 0
        elif color_mode == "face_map":
            if material.map is None:
                raise ValueError("Cannot apply colormap is no material.map is set.")
            self["color_mode"] = "face_map"
            self["color_buffer_channels"] = 0
        else:
            raise RuntimeError(f"Unknown color_mode: '{color_mode}'")

        # Optimization: when the line is opaque, has a uniform color, and no dashing,
        # it can be rendered pretty safely without joins. I *think* this is faster,
        # because a lot of logic related joins becomes simpler. However, the miters
        # result in extra fragments that need to be processed, so we'd need to do
        # some benchmarks to be sure.
        # if (
        #     self["color_mode"] == "uniform"
        #     and not self["dashing"]
        #     and material.alpha_method == "opaque"
        #     and not_using_colors_that_may_have_alpha
        # ):
        #     # self["line_type"] = "quickline"

        # Handle looping. The line_loop_buffer is one larger to enable looping the last point.
        self._loop_ranges_hash = None
        self._loop_ranges = []
        self._baked_loop_ranges = None
        if material.loop:
            self["loop"] = True
            self.line_loop_buffer = Buffer(
                np.zeros((geometry.positions.nitems + 1,), np.uint32)
            )
            self.needs_bake_function = True

        # Handle dashing
        if material.dash_pattern:
            # Set dash props
            self["dashing"] = True
            self["dash_pattern"] = tuple(wobject.material.dash_pattern)
            self["dash_count"] = len(wobject.material.dash_pattern) // 2
            # For line segments we can calculate the distance between nodes in the shader.
            # For normal lines, we need a cumulative distance.
            if not isinstance(material, LineSegmentMaterial):
                self.needs_bake_function = True
                self._cumdist_hash = None
                # Like the loop buffer, this buffer is one larger when looping, so
                # that the node that closes a loop can store the cumulative distance
                # of the full loop (see _bake_line_distance).
                self.line_distance_buffer = Buffer(
                    np.zeros(
                        (geometry.positions.nitems + int(self["loop"]),), np.float32
                    )
                )

    def bake_function(self, wobject, camera, logical_size):
        if hasattr(self, "line_loop_buffer"):
            self._bake_line_loops(wobject)
        if hasattr(self, "line_distance_buffer"):
            self._bake_line_distance(wobject, camera, logical_size)

    def _get_loop_ranges(self, positions_buffer):
        """Get the loops that the positions (in the current draw range) represent.

        Returns a list of ``(i_first, i_last, i_connector)`` tuples, with absolute
        node indices. The connector is the node that closes the loop; it is the nan
        node that follows the loop, or the (virtual) node just past the draw range.
        """
        # Early exit?
        loop_hash = (id(positions_buffer), positions_buffer.rev)
        if loop_hash == self._loop_ranges_hash:
            return self._loop_ranges
        self._loop_ranges_hash = loop_hash

        r_offset, r_size = positions_buffer.draw_range
        positions_array = positions_buffer.data

        # Get indices of points that are nan
        (nan_indices,) = np.where(
            np.isnan(positions_array[r_offset : r_offset + r_size]).any(axis=1)
        )

        # Each stretch of at least 3 non-nan nodes is a loop. Note that the last
        # stretch ends at the end of the draw range, and that the comparison with
        # n_nodes makes sure that a trailing nan node does not produce a loop.
        loop_ranges = []
        i1 = r_offset - 1
        for i2 in [*(nan_indices + r_offset), r_offset + r_size]:
            n_nodes = i2 - i1 - 1
            if n_nodes >= 3:
                loop_ranges.append((i1 + 1, i2 - 1, i2))
            i1 = i2

        self._loop_ranges = loop_ranges
        return loop_ranges

    def _bake_line_loops(self, wobject):
        # Early exit? Note that _get_loop_ranges returns the same list object
        # for as long as the positions have not changed.
        positions_buffer = wobject.geometry.positions
        loop_ranges = self._get_loop_ranges(positions_buffer)
        if loop_ranges is self._baked_loop_ranges:
            return
        self._baked_loop_ranges = loop_ranges

        # Get arrays
        loop_buffer = self.line_loop_buffer
        r_offset, r_size = positions_buffer.draw_range
        loop_array = loop_buffer.data

        is_first = 0x10000000
        is_last = 0x20000000
        is_connector = 0x30000000

        # Mark the loop nodes in the loop array
        loop_array[r_offset : r_offset + r_size + 1] = 0
        for i_first, i_last, i_connector in loop_ranges:
            n_nodes = i_last - i_first + 1
            loop_array[i_first] = is_first + n_nodes
            loop_array[i_last] = is_last + n_nodes
            loop_array[i_connector] = is_connector + n_nodes

        loop_buffer.update_range(r_offset, r_size + 1)

    def _bake_line_distance(self, wobject, camera, logical_size):
        # Prepare
        positions_buffer = wobject.geometry.positions
        r_offset, r_size = positions_buffer.draw_range

        # Prepare arrays
        positions_array = positions_buffer.data[r_offset : r_offset + r_size]
        distance_array = self.line_distance_buffer.data[r_offset : r_offset + r_size]

        finites = np.isfinite(positions_array).all(axis=1)
        has_non_finites = not finites.all()

        # Get vertices in the appropriate coordinate frame
        if wobject.material.thickness_space == "model":
            # Skip this step if the position data has not changed
            cumdist_hash = (id(positions_buffer), positions_buffer.rev)
            if cumdist_hash == self._cumdist_hash:
                return
            self._cumdist_hash = cumdist_hash
            vertex_array = positions_array
        else:
            # Prep
            if has_non_finites:
                positions_array_sub = positions_array[finites, :]
            else:
                positions_array_sub = positions_array
            # Transform
            if wobject.material.thickness_space == "world":
                vertex_array_sub = la.vec_transform(
                    positions_array_sub, wobject.world.matrix
                )
            else:  # wobject.material.thickness_space == "screen":
                xyz = la.vec_transform(
                    positions_array_sub, camera.camera_matrix @ wobject.world.matrix
                )
                vertex_array_sub = xyz[:, :2] * (0.5 * np.array(logical_size))
            # Fix up. Note that the transformed array is 2D for screen space and 3D
            # for world space, hence taking the number of columns from the result.
            if has_non_finites:
                vertex_array = np.full(
                    (len(positions_array), vertex_array_sub.shape[1]),
                    np.nan,
                    np.float32,
                )
                vertex_array[finites] = vertex_array_sub
            else:
                vertex_array = vertex_array_sub

        # Calculate distances
        distances = np.linalg.norm(vertex_array[1:] - vertex_array[:-1], axis=1)
        distances[~np.isfinite(distances)] = 0.0

        # Store cumulatives
        distance_array[0] = 0.0
        np.cumsum(distances, out=distance_array[1:])

        # Restart the cumulative distance at the beginning of each line piece, so
        # that a piece does not inherit the accumulated distance of the pieces
        # before it. Otherwise the dash phase of each successive piece is offset by
        # the total length of all preceding pieces, which makes the dashes of the
        # later pieces race ahead as soon as anything (e.g. the zoom level) changes
        # that length. This matches how SVG restarts its dashes at each subpath.
        if has_non_finites:
            # A node starts a piece if it is finite and the node before it is not.
            piece_starts = np.empty(len(distance_array), bool)
            piece_starts[0] = True
            np.logical_and(finites[1:], ~finites[:-1], out=piece_starts[1:])
            # The cumdist is non-decreasing, so a running maximum of the cumdist at
            # the piece starts (and zero elsewhere) gives, for each node, the cumdist
            # at the start of its piece.
            piece_offsets = np.where(piece_starts, distance_array, 0.0)
            np.maximum.accumulate(piece_offsets, out=piece_offsets)
            distance_array -= piece_offsets

        # For looping lines, the connecting node (the one that closes the loop)
        # stores the cumulative distance of the *closed* loop. Without this, the
        # shader would derive the length of the closing segment from the cumdist
        # of the first node, i.e. it would measure that one segment as if it spans
        # the whole loop, making its dashes much denser. See gh-1103.
        if self["loop"]:
            full_distance_array = self.line_distance_buffer.data
            for i_first, i_last, i_connector in self._get_loop_ranges(positions_buffer):
                closing_distance = np.linalg.norm(
                    vertex_array[i_last - r_offset] - vertex_array[i_first - r_offset]
                )
                if not np.isfinite(closing_distance):
                    closing_distance = 0.0
                full_distance_array[i_connector] = (
                    full_distance_array[i_last] + closing_distance
                )
            r_size += 1  # the connector of the last loop can sit just past the range

        # Mark that the data has changed
        self.line_distance_buffer.update_range(r_offset, r_size)

    def get_bindings(self, wobject, shared, scene):
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
        if positions1.data is None:
            pass  # assume the user knows that it must be 3D vertices
        elif positions1.data.shape[1] != 3:
            raise ValueError(
                "For rendering (thick) lines, the geometry.positions must be Nx3."
            )

        uniform_buffer = Buffer(
            array_from_shadertype(renderer_uniform_type), force_contiguous=True
        )
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
            bindings.append(
                Binding("s_texcoords", rbuffer, geometry.texcoords, "VERTEX")
            )
            bindings.extend(
                self.define_generic_colormap(material.map, geometry.texcoords)
            )

        # Need a buffer for the loop and/or cumdist?
        if hasattr(self, "line_loop_buffer"):
            bindings.append(Binding("s_loop", rbuffer, self.line_loop_buffer, "VERTEX"))
        if hasattr(self, "line_distance_buffer"):
            bindings.append(
                Binding("s_cumdist", rbuffer, self.line_distance_buffer, "VERTEX")
            )

        bindings = {i: b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)

        # Instanced lines have an extra storage buffer that we add manually
        bindings1 = {}  # non-auto-generated bindings
        if self["instanced"]:
            bindings1[0] = Binding(
                "s_instance_infos", rbuffer, wobject.instance_buffer, "VERTEX"
            )

        return {
            0: bindings,
            1: bindings1,
        }

    def get_pipeline_info(self, wobject, shared):
        # Cull backfaces so that overlapping faces are not drawn.
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_strip,
            "cull_mode": wgpu.CullMode.none,
        }

    def _get_n(self, positions):
        offset, size = positions.draw_range
        if self["loop"]:
            size += 1
        return offset * 6, size * 6

    def get_render_info(self, wobject, shared):
        # Determine how many vertices are needed
        offset, size = self._get_n(wobject.geometry.positions)
        inst_offset, inst_size = 0, 1
        if self["instanced"]:
            inst_offset, inst_size = wobject.instance_buffer.draw_range
        return {
            "indices": (size, inst_size, offset, inst_offset),
        }

    def get_code(self):
        return load_wgsl("line.wgsl")


@register_wgpu_render_function(Line, LineDebugMaterial)
class LineDebugShader(LineShader):
    def __init__(self, wobject):
        super().__init__(wobject)

        self["debug"] = True


@register_wgpu_render_function(Line, LineSegmentMaterial)
class LineSegmentShader(LineShader):
    """This shader is baded on the normal line shader, but it does not draw joins.
    Still needs 6 vertices in for nodes that have a cap on each side.
    """

    def __init__(self, wobject):
        super().__init__(wobject)
        self["line_type"] = "segment"


@register_wgpu_render_function(Line, LineInfiniteSegmentMaterial)
class LineInfiniteSegmentShader(LineShader):
    """Shader to draw infinite line segments. Since the line's ends are always off-screen, there is no need to draw caps."""

    def __init__(self, wobject):
        super().__init__(wobject)
        material = wobject.material
        self["line_type"] = "infsegment"
        self["start_is_infinite"] = material.start_is_infinite
        self["end_is_infinite"] = material.end_is_infinite


@register_wgpu_render_function(Line, LineArrowMaterial)
class LineArrowShader(LineShader):
    """Shader to draw arrows. This shader does not use the caps, so it could be drawn
    with less vertices, but that'd make the code more complex, so for now this is fine.
    """

    def __init__(self, wobject):
        super().__init__(wobject)
        self["line_type"] = "arrow"


# -----  shaders for thin lines


@register_wgpu_render_function(Line, LineThinMaterial)
class ThinLineShader(LineShader):
    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)
        self["aa"] = False  # no aa with thin lines
        if self["color_mode"] in ("face", "face_map"):
            raise RuntimeError("Face coloring not supported for thin lines.")

    def get_bindings(self, wobject, shared, scene):
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
            bindings.append(
                Binding("s_texcoords", rbuffer, geometry.texcoords, "VERTEX")
            )
            bindings.extend(
                self.define_generic_colormap(material.map, geometry.texcoords)
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
        offset, size = wobject.geometry.positions.draw_range
        return {
            "indices": (size, 1, offset, 0),
        }

    def get_code(self):
        return """//wgsl

        {$ include 'pygfx.std.wgsl' $}

        struct VertexInput {
            @builtin(vertex_index) index : u32,
        };

        @vertex
        fn vs_main(in: VertexInput) -> Varyings {

            let i0 = i32(in.index);

            let raw_pos = nonlinear_transform(load_s_positions(i0));
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

        @fragment
        fn fs_main(varyings: Varyings) -> FragmentOutput {
            {$ include 'pygfx.clipping_planes.wgsl' $}

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

            do_alpha_test(opacity);

            var out: FragmentOutput;
            out.color = out_color;
            return out;
        }
        """


@register_wgpu_render_function(Line, LineThinSegmentMaterial)
class ThinLineSegmentShader(ThinLineShader):
    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.line_list,
            "cull_mode": wgpu.CullMode.none,
        }
