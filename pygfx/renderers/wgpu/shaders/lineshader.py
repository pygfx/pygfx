""" pygfx line shader. See line.wgsl for details.
"""

import wgpu  # only for flags/enums
import numpy as np
import pylinalg as la

from ....utils import array_from_shadertype
from ....resources import Buffer
from ....objects import Line
from ....materials._line import (
    LineMaterial,
    LineSegmentMaterial,
    LineArrowMaterial,
    LineThinMaterial,
    LineThinSegmentMaterial,
    LineDebugMaterial,
)

from .. import (
    register_wgpu_render_function,
    BaseShader,
    Binding,
    RenderMask,
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

        self["line_type"] = "line"
        self["dashing"] = False
        self["thickness_space"] = material.thickness_space
        self["aa"] = material.aa
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

        # Optimization: when the line is opaque, has a uniform color, and no dashing,
        # it can be rendered pretty safely without joins. I *think* this is faster,
        # because a lot of logic related joins becomes simpler. However, the miters
        # result in extra fragments that need to be processed, so we'd need to do
        # some benchmarks to be sure.
        # if (
        #     self["color_mode"] == "uniform"
        #     and not self["dashing"]
        #     and not material.is_transparent
        #     and not material.color_is_transparent
        # ):
        #     # self["line_type"] = "quickline"

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
                self._positions_hash = None
                self.line_distance_buffer = Buffer(
                    np.zeros((geometry.positions.nitems,), np.float32)
                )

    def bake_function(self, wobject, camera, logical_size):
        # Prepare
        positions_buffer = wobject.geometry.positions
        r_offset, r_size = positions_buffer.draw_range

        # Prepare arrays
        positions_array = positions_buffer.data[r_offset : r_offset + r_size]
        distance_array = self.line_distance_buffer.data[r_offset : r_offset + r_size]

        # Get vertices in the appropriate coordinate frame
        if wobject.material.thickness_space == "model":
            # Skip this step if the position data has not changed
            positions_hash = (id(positions_buffer), positions_buffer.rev)
            if positions_hash == self._positions_hash:
                return
            self._positions_hash = positions_hash
            vertex_array = positions_array
        else:
            # Prep
            finites = np.isfinite(positions_array).all(axis=1)
            has_non_finites = not finites.all()
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
            # Fix up
            if has_non_finites:
                vertex_array = np.full((len(positions_array), 2), np.nan, np.float32)
                vertex_array[finites] = vertex_array_sub
            else:
                vertex_array = vertex_array_sub

        # Calculate distances
        distances = np.linalg.norm(vertex_array[1:] - vertex_array[:-1], axis=1)
        distances[~np.isfinite(distances)] = 0.0

        # Store cumulatives
        np.cumsum(distances, out=distance_array[1:])

        # Mark that the data has changed
        self.line_distance_buffer.update_range(r_offset, r_size)

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
                self.define_colormap(
                    material.map, geometry.texcoords, material.map_interpolation
                )
            )

        # Need a buffer for the cumdist?
        if hasattr(self, "line_distance_buffer"):
            bindings.append(
                Binding("s_cumdist", rbuffer, self.line_distance_buffer, "VERTEX")
            )

        bindings = {i: b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)

        return {
            0: bindings,
        }

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

        render_mask = 0
        if wobject.render_mask:
            render_mask = wobject.render_mask
        elif material.is_transparent:
            render_mask = RenderMask.transparent
        else:
            # Get what passes are needed for the color
            if self["color_mode"] == "uniform":
                if material.color_is_transparent:
                    render_mask |= RenderMask.transparent
                else:
                    render_mask |= RenderMask.opaque
            elif self["color_mode"] in ("vertex", "face"):
                if self["color_buffer_channels"] in (2, 4):
                    render_mask |= RenderMask.all
                else:
                    render_mask |= RenderMask.opaque
            elif self["color_mode"] in ("vertex_map", "face_map"):
                if self["colormap_nchannels"] in (2, 4):
                    render_mask |= RenderMask.all
                else:
                    render_mask |= RenderMask.opaque
            else:
                raise RuntimeError(f"Unexpected color mode {self['color_mode']}")
            # Need transparency for aa
            if material.aa:
                render_mask |= RenderMask.transparent

        return {
            "indices": (size, 1, offset, 0),
            "render_mask": render_mask,
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
            bindings.append(
                Binding("s_texcoords", rbuffer, geometry.texcoords, "VERTEX")
            )
            bindings.extend(
                self.define_colormap(
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
        return """//wgsl

        {$ include 'pygfx.std.wgsl' $}

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
