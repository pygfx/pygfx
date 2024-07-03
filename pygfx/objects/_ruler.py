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

    * First use ``.configure()`` to set the ruler's start- and end-point.
    * Next, the convenience functions can be used to help produce a ticks dict.
    * Finally, use ``set_ticks()`` to set the ticks and update the child world objects.
    """

    def __init__(self, *, tick_side="left", ticks_at_end_points=False):
        super().__init__()

        self.tick_side = tick_side
        self.ticks_at_end_points = ticks_at_end_points

        # Initialize dummy config
        zero = np.zeros((3,), np.float32)
        self._config = zero, zero, 0.0, 0.0, zero

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

    @property
    def line(self):
        """The line object that shows the ruler's path."""
        return self._line

    @property
    def points(self):
        """The points object that shows the ruler's tickmarks."""
        return self._points

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
        return self._tick_at_end_points

    @ticks_at_end_points.setter
    def ticks_at_end_points(self, value):
        self._ticks_at_end_points = bool(value)

    def configure(self, start_pos, end_pos, start_value=0):
        """Set the start- and end-point, optionally providing the start-value.

        Note that the ruler's transform also affects positioning, but should
        generally not be used.
        """

        # Process positions
        start_pos = np.array(start_pos, np.float32)
        if not start_pos.shape == (3,):
            raise ValueError("start_pos must be a 3-element position.")
        end_pos = np.array(end_pos, np.float32)
        if not end_pos.shape == (3,):
            raise ValueError("end_pos must be a 3-element position.")

        # Derive unit vector
        vec = end_pos - start_pos
        length = np.linalg.norm(vec)
        if length > 0.0:
            vec /= length

        # Process values
        min_value = float(start_value)
        max_value = min_value + length

        # Store
        self._config = start_pos, end_pos, min_value, max_value, vec

    def get_ticks_uniform(self, step):
        """Get ticks using a uniform step size.

        This uses the currently configured start- and end-values.

        Note that with a nonzero start_value, the first tick is likely
        not on the start position.
        """

        # Load config
        _, _, min_value, max_value, _ = self._config

        t1, t2 = self._t1_t2
        min_value = (1 - t1) * min_value + t1 * max_value
        max_value = (1 - t2) * min_value + t2 * max_value

        ref_value = max(abs(min_value), abs(max_value))

        # Apply some form of scaling
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

    def calculate_tick_step(self, camera, canvas_size, *, min_tick_dist=40):
        """Calculate an optimal step size for the ticks.

        This uses the currently configured start- and end-values. The returned
        step value is determined from the ratio of the size of the ruler in
        the world and screen, respectively.
        """

        # Load config
        start_pos, end_pos, min_value, max_value, _ = self._config

        # Yuk, but needed
        camera.update_projection_matrix()

        # Get ndc coords for begin and end pos
        positions = np.column_stack(
            [np.row_stack([start_pos, end_pos]), np.ones((2, 1), np.float64)]
        )
        ndc1, ndc2 = (camera.camera_matrix @ positions[..., None]).reshape(-1, 4)

        # Get what part of the line is visible
        t1, t2 = get_visible_part_of_line_ndc(ndc1, ndc2)
        self._t1_t2 = t1, t2

        # Get corresponding screen coordinates.
        # Fall back to full line when the selected region is empty,
        # so that we can still calculate the step size, because calling code
        # may still need it, e.g. to configure a grid.
        if t1 == t2:
            ndc_t1_2d = ndc1[:2] / ndc1[3]
            ndc_t2_2d = ndc2[:2] / ndc2[3]
        else:
            ndc_t1 = ndc1 * (1 - t1) + ndc2 * t1
            ndc_t2 = ndc1 * (1 - t2) + ndc2 * t2
            ndc_t1_2d = ndc_t1[:2] / ndc_t1[3]
            ndc_t2_2d = ndc_t2[:2] / ndc_t2[3]
        screen1 = ndc_t1_2d * np.array(canvas_size) * 0.5
        screen2 = ndc_t2_2d * np.array(canvas_size) * 0.5

        # Calculate distance on screen.
        screen_vec = screen2 - screen1
        screen_dist = np.linalg.norm(screen_vec)
        if t1 != t2:
            screen_dist /= t2 - t1

        # The orientation on screen determines the anchor for the text labels
        self._calculate_text_anchor(np.arctan2(screen_vec[1], screen_vec[0]))

        # Determine distance in world coords
        world_dist = abs(max_value - min_value)

        # Determine step
        step = 1
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

    def _calculate_text_anchor(self, angle):
        # Calculate anchor.
        # With this anchor, the text labels move smoothly without a jump to the other side, as the ruler is rotated.
        # todo: not sure how to apply it. Also may want to calculate this even if not interested in auto-ticks?
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

    def set_ticks(self, ticks):
        """Update the visual appearance of the ruler, using the given ticks.

        The ``ticks`` parameter must be a dict thet maps float values to strings.
        """

        if not isinstance(ticks, dict):
            raise TypeError("ticks must be a dict (float -> str).")

        tick_size = 5

        # Load config
        start_pos, end_pos, min_value, max_value, vec = self._config
        length = max_value - min_value

        # Get array to store positions
        n_slots = self.points.geometry.positions.nitems
        n_positions = len(ticks) + 2
        if n_positions <= n_slots <= 2 * n_positions:
            positions = self.points.geometry.positions.data
            sizes = self.points.geometry.sizes.data
            self.points.geometry.positions.update_range()
            self.points.geometry.sizes.update_range()
        else:
            new_n_slots = int(n_positions * 1.2)
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
            define_text(start_pos, f"{min_value:0.4g}")
        else:
            sizes[0] = 0
            define_text(start_pos, f"")

        # Collect ticks
        index += 1
        for value, text in ticks.items():
            rel_value = value - min_value
            if 0 <= rel_value <= length:
                pos = start_pos + vec * rel_value
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
            define_text(end_pos, f"{max_value:0.4g}")
            self._text_object_pool[1].geometry.set_text("")
            self._text_object_pool[index - 1].geometry.set_text("")
        else:
            define_text(end_pos, "")

        # Apply
        self.clear()
        self.add(self._line, self._points)
        self.add(*self._text_object_pool)
