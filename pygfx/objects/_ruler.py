from math import floor, ceil, log10

import numpy as np

from ._base import WorldObject
from ._more import Line, Points
from ._text import MultiText
from ..resources import Buffer
from ..materials import LineMaterial, PointsMaterial, TextMaterial
from ..utils.compgeo import get_visible_part_of_line_ndc


class Ruler(WorldObject):
    """An object to represent a ruler with tickmarks.

    Can be used to measure distances in a scene, or as an axis in a plot.

    Usage:

    * Use the properties (most notably ``start_pos`` and ``end_pos``).
    * Call ``update()`` on each draw.
    """

    def __init__(
        self,
        *,
        start_pos=(0, 0, 0),
        end_pos=(0, 0, 0),
        start_value=0.0,
        ticks=None,
        tick_format="0.4g",
        tick_side="left",
        min_tick_distance=50,
        ticks_at_end_points=False,
    ):
        super().__init__()

        self.start_pos = start_pos
        self.end_pos = end_pos
        self.start_value = start_value
        self.ticks = ticks

        self.tick_format = tick_format
        self.tick_side = tick_side
        self.min_tick_distance = min_tick_distance
        self.ticks_at_end_points = ticks_at_end_points

        # Create a line and points object, with a shared geometry
        self._text = MultiText(material=TextMaterial(), screen_space=True)
        geometry = self._text.geometry  # has .positions buffer
        geometry.sizes = Buffer(np.zeros(geometry.positions.nitems, "f4"))
        self._line = Line(geometry, LineMaterial(color="w", thickness=2))
        self._points = Points(geometry, PointsMaterial(color="w", size_mode="vertex"))

        self.add(self._line, self._points, self._text)

        # todo: a material to draw proper tick marks

    def get_bounding_box(self):
        # A ruler does not have a bounding box by itself
        return None

    # -- Properties to easily access sub-objects

    @property
    def line(self):
        """The line object that shows the ruler's path."""
        return self._line

    @property
    def points(self):
        """The points object that shows the ruler's tickmarks."""
        return self._points

    @property
    def text(self):
        """The text object that shows the ruler's tick labels."""
        return self._text

    # Note: text should also be here eventually

    # -- Main properties

    @property
    def start_pos(self):
        """The start posision of the ruler, in model space.

        Note that the ruler's transform also affects positioning, but should
        generally not be used.
        """
        return self._start_pos

    @start_pos.setter
    def start_pos(self, pos):
        start_pos = np.array(pos, np.float64).reshape((3,))
        if start_pos.shape != (3,):
            raise ValueError("Ruler.start_pos must be a 3-element position.")
        self._start_pos = start_pos
        self._end_value = None

    @property
    def end_pos(self):
        """The end posision of the ruler, in model space."""
        return self._end_pos

    @end_pos.setter
    def end_pos(self, pos):
        end_pos = np.array(pos, np.float64).reshape((3,))
        if end_pos.shape != (3,):
            raise ValueError("Ruler.end_pos must be a 3-element position.")
        self._end_pos = end_pos
        self._end_value = None

    @property
    def start_value(self):
        """The value of the ruler at the start position (i.e. the offset)."""
        return self._start_value

    @start_value.setter
    def start_value(self, value):
        self._start_value = float(value)
        self._end_value = None

    @property
    def end_value(self):
        """The value at the end of the ruler (read-only)."""
        # Little caching mechanic. Props that affect the end_value set ._end_value to None
        if self._end_value is None:
            self._end_value = float(
                np.linalg.norm(self._end_pos - self._start_pos) + self._start_value
            )
        return self._end_value

    @property
    def ticks(self):
        """The ticks to show.

        * ``None`` for automatic ticks.
        * ``dict`` for explicit ticks. Values can be str or float. If float, they
          are formatted with ``tick_format``.
        * ``list`` / ``tuple`` / ``ndarray`` for a list of values.
        """
        return self._ticks

    @ticks.setter
    def ticks(self, ticks):
        if ticks is None:
            self._ticks = None
        elif isinstance(ticks, dict):
            # Copy the object, resolving keys and values to float and str
            self._ticks = {float(k): v for k, v in ticks.items()}
        elif isinstance(ticks, (tuple, list, np.ndarray)):
            self._ticks = [float(x) for x in ticks]
        else:
            raise TypeError("Ruler.ticks must be None, dict or list(like).")

    # -- Properties for tweaking

    @property
    def tick_format(self):
        """The format to display the tick values.

        * A string to use as the second arg in ``format()``, default "0.4g".
        * "km" to use mili/Kilo/Mega/Giga suffixes.
        * A function that maps (value, min_value, max_value) to a str.
        """
        return self._tick_format

    @tick_format.setter
    def tick_format(self, tick_format):
        if isinstance(tick_format, str):
            self._tick_format = str(tick_format)
        elif callable(tick_format):
            # Emperically check the given function
            try:
                r = tick_format(0, -1, 1)
            except TypeError as err:
                raise ValueError(
                    f"Incompatible tick_format function: {err!s}"
                ) from None
            if not isinstance(r, str):
                raise ValueError(
                    f"Incompatible tick_format function: it must return str, not {r.__class__.__name__}"
                )
            self._tick_format = tick_format

    @property
    def tick_side(self):
        """Whether the ticks are on the 'left' or 'right' of the line.

        Imagine standing on the start position, with the line in front of you.
        """
        return self._tick_side

    @tick_side.setter
    def tick_side(self, side):
        side = str(side).lower()
        if side in ("left", "right"):
            self._tick_side = side
        else:
            raise ValueError("Tick side must be 'left' or 'right'.")

    @property
    def min_tick_distance(self):
        """The minimal distance between ticks in screen pixels, when using auto-ticks."""
        return self._min_tick_dist

    @min_tick_distance.setter
    def min_tick_distance(self, value):
        value = float(value)
        if value < 0.0:
            raise ValueError("tick distance must be larger than zero.")
        self._min_tick_dist = value

    @property
    def ticks_at_end_points(self):
        """Whether to show tickmarks at the end-points."""
        return self._ticks_at_end_points

    @ticks_at_end_points.setter
    def ticks_at_end_points(self, value):
        self._ticks_at_end_points = bool(value)

    # -- Methods

    def update(self, camera, canvas_size):
        """Update the ruler.

        This must be called on every draw, right before rendering.

        Returns a dictionary with the following fields:

        * "tick_step": the calculated auto-tick-step.
        * "tick_values": the tick values that will be shown.
        """

        # Determine which part of the ruler is on screen and its length in screen pixels
        self._configure_for_screen(camera, canvas_size)

        # Anchor
        screen_vec = self._visible_part_screen_vec
        self._calculate_text_anchor(np.arctan2(screen_vec[1], screen_vec[0]))

        # Get the dict with visible ticks
        tick_auto_step = self._calculate_tick_step()
        visible_ticks = self._get_ticks_dict(tick_auto_step)

        # Update objects to show these ticks
        self._update_sub_objects(visible_ticks, tick_auto_step)

        # Return stats. This is a dict, so we can add more stuff later, if needed.
        return {
            "tick_step": tick_auto_step,
            "tick_values": list(visible_ticks.keys()),
        }

    def _configure_for_screen(self, camera, canvas_size):
        """Make the ruler aware of the camera and viewport size."""

        half_canvas_size = 0.5 * np.array(canvas_size, np.float64).reshape(1, 2)

        # Get ndc coords for begin and end pos. Use numpy broadcasting for performance and compactness.
        positions = np.column_stack(
            [
                np.vstack([self._start_pos, self._end_pos]),
                np.ones((2, 1), np.float64),
            ]
        )
        ndc_full = (camera.camera_matrix @ positions[..., None]).reshape(-1, 4)
        screen_full = (ndc_full[:, :2] / ndc_full[:, 3:4]) * half_canvas_size

        # Get what part of the line is visible
        t1, t2 = get_visible_part_of_line_ndc(ndc_full[0], ndc_full[1])

        # Get screen coords for visible selection.
        ndc_sel = np.array(
            [
                ndc_full[0] * (1 - t1) + ndc_full[1] * t1,
                ndc_full[0] * (1 - t2) + ndc_full[1] * t2,
            ]
        )
        screen_sel = (ndc_sel[:, :2] / ndc_sel[:, 3:4]) * half_canvas_size

        # Store values
        self._screen_vec = screen_full[1] - screen_full[0]
        start_value, end_value = self._start_value, self.end_value
        self._visible_part_coords = t1, t2
        self._visible_part_values = (
            start_value * (1.0 - t1) + end_value * t1,
            start_value * (1.0 - t2) + end_value * t2,
        )
        self._visible_part_screen_vec = screen_sel[1] - screen_sel[0]

    def _calculate_text_anchor(self, angle):
        """Calculate the best place to anchor the text labels.
        With this anchor, the text labels move smoothly without a jump
        to the other side, as the ruler is rotated.
        """
        if self._tick_side == "left":
            if abs(angle) <= 0.25 * np.pi:
                self._text_anchor = "bottom-center"
                self._text_anchor_offset = 5
            elif abs(angle) >= 0.75 * np.pi:
                self._text_anchor = "top-center"
                self._text_anchor_offset = 5
            elif angle < 0:
                self._text_anchor = "middle-left"
                self._text_anchor_offset = 10
            else:
                self._text_anchor = "middle-right"
                self._text_anchor_offset = 10
        else:
            if abs(angle) <= 0.25 * np.pi:
                self._text_anchor = "top-center"
                self._text_anchor_offset = 5
            elif abs(angle) >= 0.75 * np.pi:
                self._text_anchor = "bottom-center"
                self._text_anchor_offset = 5
            elif angle < 0:
                self._text_anchor = "middle-right"
                self._text_anchor_offset = 10
            else:
                self._text_anchor = "middle-left"
                self._text_anchor_offset = 10

    def _get_ticks_dict(self, tick_auto_step):
        """Get a tick-dict, derived from the user-given tick value,
        and constrained to the visual part of the ruler.
        """

        min_value, max_value = self._visible_part_values
        tick_format = self._tick_format

        def default_tick_format_func(val, min_val, max_val):
            if val == 0:
                return "0"
            else:
                return format(val, tick_format)

        # Select funtion to format the tick values
        if tick_format == "km":
            tick_format_func = kmg_tick_format_func
        elif isinstance(tick_format, str):
            tick_format_func = default_tick_format_func
        elif callable(tick_format):
            tick_format_func = tick_format
        else:  # Fallback
            tick_format = "0.4g"
            tick_format_func = default_tick_format_func

        # Triage based on what ticks are given
        ticks = self._ticks
        if ticks is None:
            # Auto-ticks
            tick_values = self._get_ticks_uniform(min_value, max_value, tick_auto_step)
            return {t: tick_format_func(t, min_value, max_value) for t in tick_values}

        elif isinstance(ticks, list):
            # A sequence of ticks
            return {
                t: tick_format_func(t, min_value, max_value)
                for t in ticks
                if min_value <= t <= max_value
            }

        else:  # isinstance(ticks, dict):
            # A dict with specified ticks, values can be str or float
            result = {}
            for t, v in ticks.items():
                if min_value <= t <= max_value:
                    if isinstance(v, (float, int)):
                        v = tick_format_func(v, min_value, max_value)
                    elif not isinstance(v, str):
                        v = str(v)
                    result[t] = v
            return result

    def _calculate_tick_step(self):
        """Calculate the tick step from the min_tick_distance."""

        min_tick_dist = self._min_tick_dist

        # Determine distances for visible selection
        world_dist = self._visible_part_values[1] - self._visible_part_values[0]
        screen_dist = float(np.linalg.norm(self._visible_part_screen_vec))

        # Fall back to full size if selection is zero. This way, the
        # value of step still makes sense, even when the ruler itself
        # is not on screen, and calling code may still use it to e.g.
        # configure a grid. Account for roundoff errors resulting in a nonzero value.
        if screen_dist < 1e-9:
            world_dist = self.end_value - self._start_value
            screen_dist = np.linalg.norm(self._screen_vec)

        # Determine step
        step = 0
        if world_dist > 0 and screen_dist > 0:
            scale = screen_dist / world_dist
            approx_step = min_tick_dist / scale
            power10 = 10 ** floor(log10(approx_step))
            for i in (1, 2, 2.5, 5, 10):
                maybe_step = i * power10
                if maybe_step > approx_step:
                    step = maybe_step
                    break

        return step

    def _get_ticks_uniform(self, min_value, max_value, step):
        """Get a uniformly distributed set of ticks."""

        if not step:
            return []

        first_tick = ceil(min_value / step) * step
        last_tick = floor(max_value / step) * step

        ticks = []
        t = first_tick
        while t <= last_tick:
            ticks.append(t)
            t += step

        return ticks

    def _update_sub_objects(self, ticks, tick_auto_step):
        """Update the sub-objects to show the given ticks."""
        assert isinstance(ticks, dict)

        tick_size = 5

        # Load config
        start_pos = self._start_pos
        end_pos = self._end_pos
        start_value = self._start_value
        end_value = self.end_value

        # Derive some more variables
        length = end_value - start_value
        vec = end_pos - start_pos
        if length:
            vec /= length

        # Get number of positions that we need
        n_positions = len(ticks) + 2

        # Apply anchor props
        if self._text._anchor != self._text_anchor:
            self._text.anchor = self._text_anchor
        if self._text._anchor_offset != self._text_anchor_offset:
            self._text.anchor_offset = self._text_anchor_offset

        # Get the geometry to provide us with enough slots. Keep sizes array in sync
        self._text.set_text_block_count(n_positions)
        positions_buffer = self._text.geometry.positions
        sizes_buffer = self._text.geometry.sizes
        if sizes_buffer.nitems != positions_buffer.nitems:
            sizes_buffer = self._text.geometry.sizes = Buffer(
                np.zeros(positions_buffer.nitems, "f4")
            )

        # Get arrays / list that we can write to
        positions = positions_buffer.data
        sizes = sizes_buffer.data
        text_blocks = self._text._text_blocks

        # Apply start point
        index = 0
        positions[index] = start_pos
        if self._ticks_at_end_points:
            sizes[index] = tick_size
            text_blocks[index].set_text(f"{self._start_value:0.4g}")
        else:
            sizes[index] = 0
            text_blocks[index].set_text("")

        # Collect ticks
        index += 1
        for value, text in ticks.items():
            pos = start_pos + vec * (value - start_value)
            positions[index] = pos
            sizes[index] = tick_size
            text_blocks[index].set_text(text)
            index += 1

        # Apply end point
        positions[index] = end_pos
        if self._ticks_at_end_points:
            sizes[index] = tick_size
            text_blocks[index].set_text(f"{end_value:0.4g}")
        else:
            sizes[index] = 0
            text_blocks[index].set_text("")

        # Hide the ticks close to the ends?
        if self._ticks_at_end_points and ticks:
            tick_values = list(ticks.keys())
            if abs(tick_values[0] - start_value) < 0.5 * tick_auto_step:
                text_blocks[1].set_text("")
            if abs(tick_values[-1] - end_value) < 0.5 * tick_auto_step:
                text_blocks[index - 1].set_text("")

        # Make sure that the subset is drawn, and that the buffers are synced
        positions[n_positions:] = (
            np.nan
        )  # prevent a partial join to be drawn in the line
        if positions_buffer.draw_range[1] != n_positions:
            positions_buffer.draw_range = 0, n_positions
        positions_buffer.update_full()
        sizes_buffer.update_full()


# ---- Helper functions


def kmg_tick_format_func(val, min_val, max_val):
    ref_value = max(abs(min_val), abs(max_val))
    if ref_value >= 10_000_000_000:
        mult, unit = 1 / 1_000_000_000, "G"
    elif ref_value >= 10_000_000:
        mult, unit = 1 / 1_000_000, "M"
    elif ref_value >= 10000:
        mult, unit = 1 / 1000, "K"
    elif ref_value < 0.0001:
        mult, unit = 1_000_000, "u"
    elif ref_value < 0.1:
        mult, unit = 1000, "m"
    else:
        mult, unit = 1, ""
    if val == 0:
        return "0"
    else:
        return format(mult * val, "0.4g") + unit
