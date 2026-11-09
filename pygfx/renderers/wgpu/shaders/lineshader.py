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
        self["dash_fit"] = "exact"
        self["thickness_space"] = material.thickness_space
        self["dash_scaling"] = "continuous"
        self["cumdist_space"] = material.thickness_space
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
                self._dash_levels = None
                self._dash_units = None
                self._closing_dash_units = None
                # Like the loop buffer, this buffer is one larger when looping, so
                # that the node that closes a loop can store the cumulative distance
                # of the full loop (see _bake_line_distance).
                n_cumdist = geometry.positions.nitems + int(self["loop"])
                self.line_distance_buffer = Buffer(np.zeros((n_cumdist,), np.float32))
                # How much each piece's gaps are widened to make the pattern fit
                # it, for `dash_fit="spread"`. One per node, constant within a
                # piece. It is always present and always 1.0 unless spreading,
                # rather than templated, so that changing `dash_fit` does not
                # recompile the shader.
                self.line_gap_scale_buffer = Buffer(np.ones((n_cumdist,), np.float32))
                # A quantized dash pattern is baked in model space, so that it is
                # anchored to the object and cannot slide; only its period follows
                # the view scale, in powers of two. This only has an effect when
                # the pattern is sized in screen space, because in the other
                # thickness spaces the pattern is anchored to the object already.
                if (
                    material.dash_scaling == "quantized"
                    and material.thickness_space == "screen"
                ):
                    self["dash_scaling"] = "quantized"
                    self["cumdist_space"] = "model"

                # Only "spread" needs the gap widening in the shader. Keeping it
                # out of the other modes means their WGSL is byte-for-byte what
                # it was, so their rendered output is too.
                self["dash_fit"] = material.dash_fit

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

    def _get_model_units_per_pixel(self, wobject, camera, logical_size, positions):
        """How many model units one logical pixel spans, at the middle of the line.

        This is the same trick the shader uses for `thickness_ratio` (see
        line.wgsl): take a reference point, shift it by one logical pixel on
        screen, transform it back, and measure how far it moved in model space.
        Under a perspective camera the answer varies along the line, which is
        why the dash scale is taken per segment rather than once for the object.
        """
        finites = positions[np.isfinite(positions).all(axis=1)]
        if len(finites) == 0:
            return None
        reference = 0.5 * (finites.min(axis=0) + finites.max(axis=0))

        matrix = camera.camera_matrix @ wobject.world.matrix
        try:
            matrix_inv = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return None

        # One logical pixel, in ndc. The bake maps ndc to logical pixels with
        # `0.5 * logical_size`, so a pixel is `2 / logical_size` of ndc.
        ndc = la.vec_transform(reference, matrix)
        pixel_ndc = 2.0 / np.maximum(1.0, np.asarray(logical_size, float))

        offset_x = np.array([pixel_ndc[0], 0.0, 0.0])
        offset_y = np.array([0.0, pixel_ndc[1], 0.0])
        origin = la.vec_transform(ndc, matrix_inv)
        shifted_x = la.vec_transform(ndc + offset_x, matrix_inv)
        shifted_y = la.vec_transform(ndc + offset_y, matrix_inv)
        # Average the two, so that anisotropic scaling lands in the middle
        units_per_pixel = 0.5 * (
            np.linalg.norm(shifted_x - origin) + np.linalg.norm(shifted_y - origin)
        )

        if not np.isfinite(units_per_pixel) or units_per_pixel <= 0:
            return None
        return float(units_per_pixel)

    def _get_screen_positions(self, wobject, camera, logical_size):
        """The nodes of the drawn range, in logical pixels.

        The same transform the continuous path bakes its cumdist from. The
        quantized path needs it too, but only to measure how many pixels each
        segment covers, which is what sets that segment's dash scale.
        """
        positions_buffer = wobject.geometry.positions
        r_offset, r_size = positions_buffer.draw_range
        positions = positions_buffer.data[r_offset : r_offset + r_size]
        xyz = la.vec_transform(positions, camera.camera_matrix @ wobject.world.matrix)
        return xyz[:, :2] * (0.5 * np.array(logical_size))

    def _get_dash_units(self, material, model_distances, screen_distances):
        """The length of one dash unit, in model units, for each segment.

        The pattern wants one dash unit to cover `material.thickness` logical
        pixels. How many model units that is differs from segment to segment
        the moment the line has any extent along the view direction: a segment
        running away from the camera covers fewer pixels per model unit than
        one across the view. Taking one factor for the whole object -- which is
        what this did at first -- gives the foreshortened segments the wrong
        number of dashes, badly wrong once an object is turned edge-on.

        Snapping that to a power of `material.dash_scale_step`, per segment, is
        what makes the dashes hold still: for a whole-number step the dash
        starts of one level are a subset of the next level's, so a change of
        level splits each dash rather than moving it. Under a zoom every
        segment's scale changes by the same factor, so the levels step together
        and nothing slides; under a rotation they change by different factors,
        which is exactly when the dashes *should* be redistributed.

        The step is a continuous dial between the two behaviours. At 1 there is
        no snapping at all and each segment's size follows the view exactly,
        which reproduces `dash_scaling='continuous'` for *any* geometry rather
        than only for geometry at a constant depth from the camera.

        The snap has hysteresis, i.e. it is a Schmitt trigger: a level is kept
        until the ideal level is clearly past the boundary. Without it a segment
        sitting near a boundary would flip between two levels from frame to
        frame and its dashes would stutter between splitting and merging. The
        state is per segment, so it is dropped when the segment count changes.
        """
        thickness = material.thickness
        with np.errstate(divide="ignore", invalid="ignore"):
            # Model units per logical pixel, along each segment.
            units_per_pixel = model_distances / screen_distances
        # A segment of zero screen length (edge-on, or degenerate) has no
        # defined scale; it also has no dashes to get wrong.
        bad = ~np.isfinite(units_per_pixel) | (units_per_pixel <= 0)
        units_per_pixel = np.where(bad, 1.0, units_per_pixel)
        nominal = max(thickness, 1e-9) * units_per_pixel

        step = material.dash_scale_step
        if step <= 1.0:
            # The limit of the snapping below, and the behaviour of
            # dash_scaling='continuous': each size follows the view exactly.
            self._dash_levels = None
            return nominal

        # Where in the step the size range sits. The ratio of the snapped size
        # to the nominal one is step**(level - ideal), and level is the floor of
        # (ideal + bias), so the ratio spans [step**(bias-1), step**bias]: the
        # bias is the log of the largest scale allowed. None means centred.
        max_scale = material.dash_max_scale
        if max_scale is None:
            bias = 0.5
        else:
            bias = np.log(min(max(max_scale, 1.0), step)) / np.log(step)
        with np.errstate(divide="ignore", invalid="ignore"):
            biased = np.log(nominal) / np.log(step) + bias
        biased = np.where(np.isfinite(biased), biased, 0.0)

        previous = self._dash_levels
        if previous is None or len(previous) != len(biased):
            levels = np.floor(biased)
        else:
            hysteresis = material.dash_scale_hysteresis
            keep = (previous - hysteresis <= biased) & (
                biased < previous + 1.0 + hysteresis
            )
            levels = np.where(keep, previous, np.floor(biased))

        self._dash_levels = levels
        return step**levels

    def _fit_dash_periods(
        self, wobject, positions_buffer, r_offset, r_size, piece_starts, phase_per_unit
    ):
        """Scale each line piece so that a whole number of dash periods spans it.

        Without this the pattern simply stops wherever the piece ends, which
        leaves a closed piece with a visible seam: the pattern comes back round
        to its start mid-period, so the stroke that closes the loop is the wrong
        length or lands on top of the first one. Scaling the phase of a piece by
        the small factor that makes it a whole number of periods removes that,
        at the cost of a dash size that is no longer exactly the one asked for.

        Open and closed pieces are fitted to different targets. A closed piece
        wants a whole number of periods, so that the end of the pattern meets
        its beginning. An open piece wants a whole number of periods *less its
        final gap*, so that it both begins and ends with a full stroke rather
        than trailing off into empty space.

        The count is snapped differently under the two scalings, and for the
        same reason the level is snapped in `_get_dash_unit`. Under 'continuous'
        the nearest whole number is what is wanted. Under 'quantized' the count
        is snapped to a power of `dash_scale_step` instead, because the whole
        point of that mode is that a level change splits each dash rather than
        moving it: a count on the power ladder is divisible by the step, so it
        keeps that property, whereas the nearest whole number does not (7
        periods cannot split into 14 and then back into 7 via a rounded halving).
        """
        material = wobject.material
        pattern = material.dash_pattern
        period = sum(pattern)
        if period <= 0:
            return
        last_gap = pattern[-1]

        data = self.line_distance_buffer.data
        piece_bounds = np.append(np.flatnonzero(piece_starts), r_size)
        starts, stops = piece_bounds[:-1], piece_bounds[1:]

        # The phase span of a piece sits at its last node. Nodes across a nan
        # carry the previous cumdist forward (their distances are zeroed), so
        # for an open piece that is the last node before the next piece starts.
        # A loop instead ends at its connector, which holds the length of the
        # *closed* loop, and which is the first nan node after the loop.
        span_index = stops - 1
        is_closed = np.zeros(len(starts), bool)
        if self["loop"]:
            for i_first, _i_last, i_connector in self._get_loop_ranges(
                positions_buffer
            ):
                (piece,) = np.nonzero(starts == i_first - r_offset)
                span_index[piece] = i_connector - r_offset
                is_closed[piece] = True

        span = data[r_offset + span_index] * phase_per_unit

        # How many periods the piece would like to hold, given its target.
        with np.errstate(divide="ignore", invalid="ignore"):
            wanted = np.where(is_closed, span, span + last_gap) / period
        if self["dash_scaling"] == "quantized" and self._dash_levels is not None:
            step = material.dash_scale_step
            with np.errstate(divide="ignore", invalid="ignore"):
                count = step ** np.round(np.log(wanted) / np.log(step))
        else:
            count = np.round(wanted)
        count = np.maximum(np.nan_to_num(count, nan=1.0), 1.0)

        if material.dash_fit == "spread":
            return self._spread_dash_gaps(
                material, data, r_offset, r_size, starts, stops, span, is_closed
            )

        target = count * period - np.where(is_closed, 0.0, last_gap)
        # A pattern whose period is all gap (e.g. [0, 4]) has nothing to end on,
        # so there is no sensible open-piece target; fall back to whole periods.
        target = np.where(target > 0, target, count * period)

        # A zero-length piece has no phase to scale, and nothing to show either.
        scale = np.where(span > 0, target / np.where(span > 0, span, 1.0), 1.0)
        data[r_offset : r_offset + r_size] *= np.repeat(scale, stops - starts)

    def _spread_dash_gaps(
        self, material, data, r_offset, r_size, starts, stops, span, is_closed
    ):
        """Fit by widening the gaps, leaving the strokes exactly as asked.

        `stretch` scales the whole pattern, so reaching the *nearest* whole
        number of periods often means compressing a piece: its strokes come out
        shorter and closer together than `dash_pattern` asked for. Here the
        pattern is a floor instead. The count is rounded *down*, so the piece
        always holds at least as much room as the pattern wants, and the slack
        left over is put into the gaps, which is the only part that may grow.

        Solving for the gap factor k, with T the stroke total, G the gap total
        and gl the final gap of one period:

            closed piece:  span = n * (T + k * G)          -> k = (span/n - T) / G
            open piece:    span = n * (T + k * G) - k * gl -> k = (span - n*T) / (n*G - gl)

        and k >= 1 exactly when n <= (span + gl) / period, hence the floor. The
        cumdist is left alone; k travels to the fragment shader in its own
        buffer and widens `gap_sizes` there.
        """
        pattern = material.dash_pattern
        strokes, gaps = pattern[::2], pattern[1::2]
        stroke_total, gap_total = sum(strokes), sum(gaps)
        last_gap = gaps[-1]

        gap_scale = self.line_gap_scale_buffer.data
        if gap_total <= 0:
            # Nothing to widen: a pattern with no gaps cannot be spread.
            gap_scale[r_offset : r_offset + r_size] = 1.0
            self.line_gap_scale_buffer.update_range(r_offset, r_size)
            return

        period = stroke_total + gap_total
        with np.errstate(divide="ignore", invalid="ignore"):
            count = np.floor((span + np.where(is_closed, 0.0, last_gap)) / period)
        count = np.maximum(np.nan_to_num(count, nan=1.0), 1.0)

        denominator = np.where(
            is_closed, count * gap_total, count * gap_total - last_gap
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            k = (span - count * stroke_total) / denominator
        # A piece too short for even one period would need k < 1, i.e. drawing
        # something smaller than asked. Leave those alone; the pattern runs off
        # the end, which is what "exact" does everywhere.
        k = np.where(np.isfinite(k), k, 1.0)
        k = np.maximum(k, 1.0)

        gap_scale[r_offset : r_offset + r_size] = np.repeat(k, stops - starts)
        self.line_gap_scale_buffer.update_range(r_offset, r_size)

    def _bake_line_distance(self, wobject, camera, logical_size):
        # Prepare
        positions_buffer = wobject.geometry.positions
        r_offset, r_size = positions_buffer.draw_range

        # Prepare arrays
        positions_array = positions_buffer.data[r_offset : r_offset + r_size]
        distance_array = self.line_distance_buffer.data[r_offset : r_offset + r_size]

        finites = np.isfinite(positions_array).all(axis=1)
        has_non_finites = not finites.all()

        # A quantized pattern is measured in model space, and then divided by the
        # length of one dash unit, so that the buffer holds the phase directly.
        # That length follows the view scale, but only in powers of two, so it
        # changes rarely: the whole bake can be skipped while the level holds.
        quantized = self["dash_scaling"] == "quantized"

        # Get vertices in the appropriate coordinate frame
        if self["cumdist_space"] == "model":
            # Skip this step if neither the positions nor the level have changed
            fit = wobject.material.dash_fit
            cumdist_hash = (
                id(positions_buffer),
                positions_buffer.rev,
                # Fitting depends on the pattern and the thickness, which the
                # plain model-space cumdist does not; without these, changing
                # either would be silently skipped by the early-out below. They
                # are left out when not fitting, so that the default path still
                # re-bakes only when the positions move.
                fit,
                *(
                    (wobject.material.dash_pattern, wobject.material.thickness)
                    if fit == "stretch"
                    else ()
                ),
            )
            if cumdist_hash == self._cumdist_hash and not quantized:
                # A quantized bake depends on the camera as well, through the
                # per-segment scale, so it cannot take this shortcut. Its own
                # early-out is below, once the levels are known.
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
            if self["cumdist_space"] == "world":
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

        # Convert model units to dash units, per segment. One dash unit spans
        # `thickness` logical pixels, which is a different number of model units
        # for a segment running away from the camera than for one across the
        # view, so the conversion cannot be a single factor for the object. It
        # is snapped to a power of `dash_scale_step`, which is what keeps the
        # dashes from sliding: a change of level splits each dash in place.
        if quantized:
            screen = self._get_screen_positions(wobject, camera, logical_size)
            screen_distances = np.linalg.norm(screen[1:] - screen[:-1], axis=1)
            screen_distances[~np.isfinite(screen_distances)] = 0.0

            # The segment that closes a loop is not one of the pairs above, and
            # is foreshortened independently of them, so it needs a scale of its
            # own. Appending them here rather than treating them separately
            # keeps one array of levels, and so one consistent hysteresis state.
            loop_ranges = (
                self._get_loop_ranges(positions_buffer) if self["loop"] else []
            )
            closing_model = np.array(
                [
                    np.linalg.norm(
                        vertex_array[i_last - r_offset]
                        - vertex_array[i_first - r_offset]
                    )
                    for i_first, i_last, _ in loop_ranges
                ],
                np.float64,
            ).reshape(-1)
            closing_screen = np.array(
                [
                    np.linalg.norm(
                        screen[i_last - r_offset] - screen[i_first - r_offset]
                    )
                    for i_first, i_last, _ in loop_ranges
                ],
                np.float64,
            ).reshape(-1)
            closing_model[~np.isfinite(closing_model)] = 0.0
            closing_screen[~np.isfinite(closing_screen)] = 0.0

            n_open = len(distances)
            units = self._get_dash_units(
                wobject.material,
                np.concatenate([distances, closing_model]),
                np.concatenate([screen_distances, closing_screen]),
            )
            self._dash_units = units[:n_open]
            self._closing_dash_units = units[n_open:]
            distances = distances / self._dash_units

        # Store cumulatives
        distance_array[0] = 0.0
        np.cumsum(distances, out=distance_array[1:])

        # Restart the cumulative distance at the beginning of each line piece, so
        # that a piece does not inherit the accumulated distance of the pieces
        # before it. Otherwise the dash phase of each successive piece is offset by
        # the total length of all preceding pieces, which makes the dashes of the
        # later pieces race ahead as soon as anything (e.g. the zoom level) changes
        # that length. This matches how SVG restarts its dashes at each subpath.
        # A node starts a piece if it is finite and the node before it is not.
        piece_starts = np.zeros(len(distance_array), bool)
        piece_starts[0] = True
        if has_non_finites:
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
            for i_loop, (i_first, i_last, i_connector) in enumerate(
                self._get_loop_ranges(positions_buffer)
            ):
                closing_distance = np.linalg.norm(
                    vertex_array[i_last - r_offset] - vertex_array[i_first - r_offset]
                )
                if not np.isfinite(closing_distance):
                    closing_distance = 0.0
                if quantized:
                    closing_distance /= self._closing_dash_units[i_loop]
                full_distance_array[i_connector] = (
                    full_distance_array[i_last] + closing_distance
                )
            r_size += 1  # the connector of the last loop can sit just past the range

        # Stretch each piece a little, so that a whole number of dash periods
        # spans it and the pattern has no seam. The cumdist is a phase, so this
        # is just a per-piece factor on the values already computed; the shader
        # is not involved. One dash unit is `thickness` of whatever the cumdist
        # is measured in, except under quantization, where the division above
        # has already put the buffer in dash units.
        if wobject.material.dash_fit != "spread":
            gap_scale = self.line_gap_scale_buffer.data
            if np.any(gap_scale[r_offset : r_offset + r_size] != 1.0):
                gap_scale[r_offset : r_offset + r_size] = 1.0
                self.line_gap_scale_buffer.update_range(r_offset, r_size)

        if wobject.material.dash_fit in ("stretch", "spread"):
            self._fit_dash_periods(
                wobject,
                positions_buffer,
                r_offset,
                r_size,
                piece_starts,
                1.0 if quantized else 1.0 / wobject.material.thickness,
            )

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
            bindings.append(
                Binding("s_gapscale", rbuffer, self.line_gap_scale_buffer, "VERTEX")
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
