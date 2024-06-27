from math import floor, ceil, log10

import numpy as np
import pylinalg as la

from ._base import WorldObject
from ._more import Line, Points, Text
from ..resources import Buffer
from ..geometries import Geometry, TextGeometry
from ..materials import LineMaterial, PointsMaterial, TextMaterial


class Ruler(WorldObject):
    """An object to represent a ruler with tickmarks.

    Can be used to measure distances in a scene, or as an axis in a plot.

    Usage:

    * First use ``.configure()`` to set the ruler's start- and end-point.
    * Next, the convenience functions can be used to help produce a ticks dict.
    * Finally, use ``set_ticks()`` to set the ticks and update the child world objects.
    """

    def __init__(self, *, tick_side="left"):
        super().__init__()

        self.tick_side = tick_side

        # Initialize dummy config
        zero = np.zeros((3,), np.float32)
        self._config = zero, zero, 0.0, 0.0, zero

        # Create a line and poins object, with a shared geometry
        geometry = Geometry(positions=np.zeros((6, 3), np.float32))
        self._line = Line(geometry, LineMaterial(color="w", thickness=2))
        # todo: a material to draw proper tick marks
        self._points = Points(geometry, PointsMaterial(color="w", size=5))
        self.add(self._line, self._points)

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

        # Determine distance in screen pixels
        # todo: can get a very small size with perspective camera.
        positions = np.row_stack([start_pos, end_pos])
        ndc_positions = la.vec_transform(positions, camera.camera_matrix)
        pixel_positions = ndc_positions[:, :2] * np.array(canvas_size)
        pixel_vec = pixel_positions[1] - pixel_positions[0]
        distance_screen = 0.5 * np.linalg.norm(pixel_vec)

        # Calculate anchor.
        # With this anchor, the text labels move smoothly without a jump to the other side, as the ruler is rotated.
        # todo: not sure how to apply it. Also may want to calculate this even if not interested in auto-ticks?
        angle = np.arctan2(pixel_vec[1], pixel_vec[0])
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

        # Determine distnce in world coords
        distance_world = abs(max_value - min_value)

        # Determine step
        step = 1
        if distance_world > 0:
            scale = distance_screen / distance_world
            approx_step = min_tick_dist / scale
            power10 = 10 ** floor(log10(approx_step))
            for i in (1, 2, 2.5, 5, 10):
                maybe_step = i * power10
                if maybe_step > approx_step:
                    step = maybe_step
                    break

        return step

    def set_ticks(self, ticks):
        """Update the visual appearance of the ruler, using the given ticks.

        The ``ticks`` parameter must be a dict thet maps float values to strings.
        """

        if not isinstance(ticks, dict):
            raise TypeError("ticks must be a dict (float -> str).")

        # Load config
        start_pos, end_pos, min_value, max_value, vec = self._config
        length = max_value - min_value

        # List of positions. The start- and end-pos are in it, as well as all ticks.
        positions = [start_pos]
        text_objects = []

        # Collect ticks
        for value, text in ticks.items():
            rel_value = value - min_value
            if 0 <= rel_value <= length:
                pos = start_pos + vec * rel_value
                positions.append(pos)
                text_ob = Text(
                    TextGeometry(
                        text,
                        screen_space=True,
                        anchor=self._text_anchor,
                        anchor_offset=self._text_anchor_offset,
                    ),
                    TextMaterial(),
                )
                text_ob.local.position = pos
                text_objects.append(text_ob)

        positions.append(end_pos)  # todo: somehow omit tickmarker for first and last

        # Update geometrty of line and points.
        self.points.geometry.positions = Buffer(np.array(positions, np.float32))

        # todo: use a pool of text objects. Or better yet, a text object supporting multiple locations
        self.clear()
        self.add(self._line, self._points)
        self.add(*text_objects)
