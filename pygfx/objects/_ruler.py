from math import floor, ceil, log10

import numpy as np

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

        # n_steps = 10
        # pos_vec = (end_pos - start_pos)
        # positions = [start_pos + pos_vec * (i / n_steps) for i in range(n_steps)]

        # Determine distance in screen pixels
        positions = np.column_stack(
            [
                np.row_stack([start_pos, 0.5 * (start_pos + end_pos), end_pos]),
                np.ones((3, 1), np.float32),
            ]
        )
        # ndc_positions = la.vec_transform(positions, camera.camera_matrix)
        ndc_positions = camera.camera_matrix @ positions[..., None]
        ndc_positions = ndc_positions.reshape(-1, 4)

        ndc1 = ndc_positions[0]
        ndc2 = ndc_positions[2]

        tx1 = binary_search_for_ndc_edge(ndc1, ndc2, -1, 0)
        tx2 = binary_search_for_ndc_edge(ndc1, ndc2, +1, 0)
        ty1 = binary_search_for_ndc_edge(ndc1, ndc2, -1, 1)
        ty2 = binary_search_for_ndc_edge(ndc1, ndc2, +1, 1)

        tx1, tx2 = min(tx1, tx2), max(tx1, tx2)
        ty1, ty2 = min(ty1, ty2), max(ty1, ty2)
        t1, t2 = max(tx1, ty1), min(tx2, ty2)

        t1, t2 = min(t1, t2), max(t1, t2)

        ndc_t1 = ndc1 * (1 - t1) + ndc2 * t1
        ndc_t2 = ndc1 * (1 - t2) + ndc2 * t2
        ndc1_2d = ndc_t1[:2] / ndc_t1[3]
        ndc2_2d = ndc_t2[:2] / ndc_t2[3]

        screen1 = ndc1_2d * np.array(canvas_size) * 0.5
        screen2 = ndc2_2d * np.array(canvas_size) * 0.5
        pixel_vec = screen2 - screen1
        distance_screen = np.linalg.norm(pixel_vec)

        if t1 != t2:
            distance_screen /= t2 - t1

        self._t1_t2 = t1, t2

        # ndc_positions = ndc_positions_normed

        # # Select largest section on screen
        # distance_to_center = np.linalg.norm(ndc_positions[:,:2], axis=1)
        # ref_index1 = np.argmin(distance_to_center)
        # if ref_index1 == 0:
        #     ref_index2 = 1
        # elif ref_index1 == n_steps - 1:
        #     ref_index2 = ref_index1 - 1
        # elif distance_to_center[ref_index1-1] < distance_to_center[ref_index1+1]:
        #     ref_index2 = ref_index1 - 1
        # else:
        #     ref_index2 = ref_index1 + 1
        # ref_index1, ref_index2 = min(ref_index1, ref_index2), max(ref_index1, ref_index2)
        # ndc_positions = ndc_positions[[ref_index1, ref_index2]]

        # # ndc_positions = ndc_positions[[0, -1]]
        # if np.any(abs(ndc_positions[:, :2]) > 1):
        #     breakpoint()
        #
        # pixel_positions = ndc_positions[:, :2] * np.array(canvas_size) * 0.5
        # pixel_vec = pixel_positions[1] - pixel_positions[0]
        # distance_screen = np.linalg.norm(pixel_vec)
        #
        # print(ndc_positions[0][0], ndc_positions[1][0])

        # # With a perspective camera, if either end is beyond the edge, the
        # distance_screen = min(distance_screen, 2 * max(canvas_size))

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
        if distance_world > 0 and distance_screen > 0:
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


def binary_search_for_ndc_edge(ndc1, ndc2, ref, dim):
    # Perform a binary search to find the t that results in the smallest
    # distance to ref. We use homogeneous ndc coords, so that they can
    # be interpolated. Note, however, that this means that the distance
    # is not linear. This means that if we sample two points, we cannot
    # simply select the half that has the smallest distance, since the
    # minimum can still be in the other half. To work around this, we
    # sample 5 locations, and select a half based on the point with
    # smallest distance.
    #
    #  |------|------|------|------|
    #  0      1      2      3      4

    # I spent some time optimizing this code. Using numpy arrays and np.argmin
    # is not faster. We can re-use 2 or 3 of the samples at each step, by using
    # tuples, copying these into the new slot is efficient.

    # Reduce the ndc positions to two scalars per position
    x1, x2 = ndc1[dim], ndc2[dim]
    w1, w2 = ndc1[3], ndc2[3]

    def new_sample(t):
        x = x1 * (1 - t) + x2 * t
        w = w1 * (1 - t) + w2 * t
        w = max(0.0, w)  # sometimes w is negative, resulting in wrong results
        return t, float(abs(ref - x / w))

    def argmin(samples):
        smallest_i = 0
        smallest_d = 1e16
        for i in range(5):
            d = samples[i][1]
            if d < smallest_d:
                smallest_d = d
                smallest_i = i
        return smallest_i

    # Produce 5 tuples (t, dist) representing the relative t-values as shown in above ascii diagram
    samples = [
        new_sample(0.0),
        new_sample(0.25),
        new_sample(0.5),
        new_sample(0.75),
        new_sample(1.0),
    ]

    # Number of iterations. This determines in how much pieces the ruler is divided.
    # With n = 12, we explore a space of over 4k samples.
    n = 12

    for _ in range(n):
        i = argmin(samples)
        if i == 0:
            # samples[0] = samples[0]
            samples[4] = samples[1]
            samples[2] = new_sample(0.5 * (samples[0][0] + samples[4][0]))
        elif i == 1:
            # samples[0] = samples[0]
            samples[4] = samples[2]
            samples[2] = samples[1]
        elif i == 2:
            samples[0] = samples[1]
            samples[4] = samples[3]
            # samples[2] = samples[2]
        elif i == 3:
            samples[0] = samples[2]
            # samples[4] = samples[4]
            samples[2] = samples[3]
        else:  # i == 4
            samples[0] = samples[3]
            # samples[4] = samples[4]
            samples[2] = new_sample(0.5 * (samples[0][0] + samples[4][0]))
        samples[1] = new_sample(0.5 * (samples[0][0] + samples[2][0]))
        samples[3] = new_sample(0.5 * (samples[2][0] + samples[4][0]))

    i = argmin(samples)
    return samples[i][0]
