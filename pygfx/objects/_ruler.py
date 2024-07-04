from math import floor, ceil, log10

import numpy as np

from ._base import WorldObject
from ._more import Line, Points, Text
from ..resources import Buffer
from ..geometries import Geometry, TextGeometry
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
        self, *, tick_side="left", ticks_at_end_points=False, min_tick_distance=40
    ):
        super().__init__()

        self.start_pos = 0, 0, 0
        self.end_pos = 0, 0, 0
        self.start_value = 0.0
        self.ticks = None

        self.tick_side = tick_side
        self.ticks_at_end_points = ticks_at_end_points
        self.min_tick_distance = min_tick_distance

        # Create a line and poins object, with a shared geometry
        geometry = Geometry(
            positions=np.zeros((6, 3), np.float32),
            sizes=np.zeros((6,), np.float32),
        )
        self._line = Line(geometry, LineMaterial(color="w", thickness=2))
        self._points = Points(geometry, PointsMaterial(color="w", size_mode="vertex"))
        self.add(self._line, self._points)
        self._text_object_pool = []

        # todo: a material to draw proper tick marks
        # todo: Text object that draws each line at a position from geometry

    # -- Properties to easily access sub-objects

    @property
    def line(self):
        """The line object that shows the ruler's path."""
        return self._line

    @property
    def points(self):
        """The points object that shows the ruler's tickmarks."""
        return self._points

    # Note: text should also be here eventually

    # -- Main properties

    @property
    def start_pos(self):
        """The start posision of the ruler as a 3-tuple or 3-element array.

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
        """The end posision of the ruler as a 3-tuple or 3-element array."""
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
            self._end_value = (
                np.linalg.norm(self._end_pos - self._start_pos) + self._start_value
            )
        return self._end_value

    @property
    def ticks(self):
        """The ticks to show.

        Can be:

        * ``None`` for automatic ticks.
        * ``dict`` for explicit ticks (float -> str).
        * ``list`` / ``tupple`` / ``ndarray`` for a list of values.

        """
        return self._ticks

    @ticks.setter
    def ticks(self, ticks):
        if ticks is None:
            self._ticks = None
        elif isinstance(ticks, dict):
            # Copy the object, resolving keys and values to float and str
            self._ticks = {float(k): str(v) for k, v in ticks.items()}
        elif isinstance(ticks, (tuple, list, np.ndarray)):
            self._ticks = [float(x) for x in ticks]
        else:
            raise TypeError("Ruler.ticks must be None, dict or list(like).")

    # -- Properties for tweaking

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
    def tick_side(self):
        """Whether the ticks are on the left or right of the line.

        Imagine standing on the start position, with the line in front of you.
        You must call `set_ticks()` for this change to take effect.
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
    def ticks_at_end_points(self):
        """Whether to show tickmarks at the end-points.

        You must call `set_ticks()` for this change to take effect.
        """
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

        # Yuk, but needed
        camera.update_projection_matrix()

        half_canvas_size = 0.5 * np.array(canvas_size, np.float64)

        # Get ndc coords for begin and end pos
        positions = np.column_stack(
            [
                np.row_stack([self._start_pos, self._end_pos]),
                np.ones((2, 1), np.float64),
            ]
        )
        ndc1, ndc2 = (camera.camera_matrix @ positions[..., None]).reshape(-1, 4)

        ndc1_2d = ndc1[:2] / ndc1[3]
        ndc2_2d = ndc2[:2] / ndc2[3]
        screen1 = ndc1_2d * half_canvas_size
        screen2 = ndc2_2d * half_canvas_size

        # Get what part of the line is visible
        t1, t2 = get_visible_part_of_line_ndc(ndc1, ndc2)

        # Get corresponding screen coordinates.
        # Fall back to full line when the selected region is empty,
        # so that we can still calculate the step size, because calling code
        # may still need it, e.g. to configure a grid.
        ndc_t1 = ndc1 * (1 - t1) + ndc2 * t1
        ndc_t2 = ndc1 * (1 - t2) + ndc2 * t2
        ndc_t1_2d = ndc_t1[:2] / ndc_t1[3]
        ndc_t2_2d = ndc_t2[:2] / ndc_t2[3]
        screen_t1 = ndc_t1_2d * half_canvas_size
        screen_t2 = ndc_t2_2d * half_canvas_size

        # Store values
        self._screen_vec = screen2 - screen1
        start_value, end_value = self._start_value, self.end_value
        self._visible_part_coords = t1, t2
        self._visible_part_values = (
            start_value * (1.0 - t1) + end_value * t1,
            start_value * (1.0 - t2) + end_value * t2,
        )
        self._visible_part_screen_vec = screen_t2 - screen_t1

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

        ticks = self._ticks
        if ticks is None:
            return self._get_ticks_uniform(min_value, max_value, tick_auto_step)
        elif isinstance(ticks, dict):
            return {x: v for x, v in ticks.items() if min_value <= x <= max_value}
        else:
            return {x: str(x) for x in ticks if min_value <= x <= max_value}

    def _calculate_tick_step(self):
        """Calculate the tick step from the min_tick_distance."""

        min_tick_dist = self._min_tick_dist

        # Determine distances for visible selection
        world_dist = self._visible_part_values[1] - self._visible_part_values[0]
        screen_dist = np.linalg.norm(self._visible_part_screen_vec)

        # Fall back to full size if selection is zero. This way, the
        # value of step still makes sense, even when the ruler itself
        # is not on screen.
        if not screen_dist:
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
            return {}

        # Apply some form of scaling
        ref_value = max(abs(min_value), abs(max_value))
        if False:  # use mk units
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
        elif False:  # use exponential notation
            pass
        else:
            mult, unit = 1, ""

        # Calculate tick values
        first_tick = ceil(min_value * mult / step) * step
        last_tick = floor(max_value * mult / step) * step
        ticks = {}
        t = first_tick
        while t <= last_tick:
            if t == 0:
                s = "0"
            else:
                s = f"{mult * t:0.4g}"
            ticks[t] = s + unit
            t += step

        return ticks

    def _update_sub_objects(self, ticks, tick_auto_step):
        """Update the sub-objects to show the given ticks."""
        assert isinstance(ticks, dict)

        tick_size = 5
        min_n_slots = 8  # todo: can be (much) higher when we use a single text object!

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

        # Get array to store positions
        n_slots = self.points.geometry.positions.nitems
        n_positions = len(ticks) + 2
        if n_positions <= n_slots <= max(min_n_slots, 2 * n_positions):
            # Re-use existing buffers
            positions = self.points.geometry.positions.data
            sizes = self.points.geometry.sizes.data
            self.points.geometry.positions.update_range()
            self.points.geometry.sizes.update_range()
        else:
            # Allocate new buffers
            new_n_slots = max(min_n_slots, int(n_positions * 1.2))
            positions = np.zeros((new_n_slots, 3), np.float32)
            sizes = np.zeros((new_n_slots,), np.float32)
            self.points.geometry.positions = Buffer(positions)
            self.points.geometry.sizes = Buffer(sizes)

        def define_text(pos, text):
            if index < len(self._text_object_pool):
                ob = self._text_object_pool[index]
            else:
                ob = Text(
                    TextGeometry("", screen_space=True),
                    TextMaterial(),
                )
                self._text_object_pool.append(ob)
            ob.geometry.anchor = self._text_anchor
            ob.geometry.anchor_offset = self._text_anchor_offset
            ob.geometry.set_text(text)
            ob.local.position = pos

        # Apply start point
        index = 0
        positions[0] = start_pos
        if self._ticks_at_end_points:
            sizes[0] = tick_size
            define_text(start_pos, f"{self._start_value:0.4g}")
        else:
            sizes[0] = 0
            define_text(start_pos, f"")

        # Collect ticks
        index += 1
        for value, text in ticks.items():
            pos = start_pos + vec * (value - start_value)
            positions[index] = pos
            sizes[index] = tick_size
            define_text(pos, text)
            index += 1

        # Handle end point
        positions[index:] = end_pos
        sizes[index:] = 0
        self._text_object_pool[index + 1 :] = []

        if self._ticks_at_end_points:
            sizes[index] = tick_size
            define_text(end_pos, f"{end_value:0.4g}")
        else:
            define_text(end_pos, "")

        # Hide the ticks close to the ends?
        if self._ticks_at_end_points and ticks:
            tick_values = list(ticks.keys())
            if abs(tick_values[0] - start_value) < 0.5 * tick_auto_step:
                self._text_object_pool[1].geometry.set_text("")
            if abs(tick_values[-1] - end_value) < 0.5 * tick_auto_step:
                self._text_object_pool[index - 1].geometry.set_text("")

        # Apply
        self.clear()
        self.add(self._line, self._points)
        self.add(*self._text_object_pool)
