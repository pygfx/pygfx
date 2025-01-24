import numpy as np


class Interpolant:
    """Abstract base class of interpolants over parametric samples.

    This code implements an abstract interpolation class Interpolant.
    It provides a generic template method to find the interval in the array of `parameter_positions`
    that the given parameter value t belongs to.

    It adopts the following optimization measures:

        * Utilizing the binary search algorithm to quickly locate the target interval.
        this reduces the time complexity of the lookup from linear to logarithmic.

        * Caching the index of the last lookup `_cached_index`.
        For continuous lookups within the same interval, the cached value can be directly used without the need for a new search.
        And when the lookup point is close to the cached index, the algorithm will scan the adjacent intervals to find the target interval.

        * Handling the boundary cases (when the lookup point is at the beginning or end of the array) with special processing
        avoiding unnecessary binary searches.

    This approach based on binary search and caching can effectively improve the efficiency of interpolation lookups.
    Time complexity is O(1) for linear access crossing at most 2 intervals
    and O(log2(n)) for random access, where n is the number of positions.

    Parameters
    ----------
    parameter_positions : array_like
        One dimensional array, typically the time or a path along a curve defined by the data.
    sample_values : array_like
        The sample values, can have any dimensionality.
    """

    def __init__(self, parameter_positions, sample_values):
        self.parameter_positions = parameter_positions
        self._cached_index = 0
        self.sample_values = sample_values

    def evaluate(self, t):
        pp = self.parameter_positions
        i1 = self._cached_index
        len_pp = len(pp)

        t1 = pp[i1] if i1 < len_pp else None
        t0 = pp[i1 - 1] if i1 > 0 else None

        # check if t is in the interval of the cached index, scan the 2 adjacent intervals at most
        if t1 is None or t >= t1:
            # scan the right side of the interval, at most 2 intervals
            for _ in range(2):
                i1 += 1
                if i1 >= len_pp:
                    if t < t0:
                        break  # break to the binary search

                    # after the end
                    i1 = len_pp
                    self._cached_index = i1
                    return self.sample_values[i1 - 1]

                t0 = t1
                t1 = pp[i1]

                if t < t1:
                    # we have arrived at the sought interval
                    self._cached_index = i1
                    return self._interpolate(i1, t0, t, t1)

            # prepare binary search on the right side of the index
            right = len_pp
        elif t0 is None or t < t0:
            global_t1 = pp[1]  # the first keyframe
            if t < global_t1:
                # when the animation is looped, scan the biggining after the end
                i1 = 2
                t0 = global_t1

            # scan the left side of the interval, at most 2 intervals
            for _ in range(2):
                i1 -= 1
                if i1 <= 0:
                    # before the start
                    self._cached_index = 0
                    return self.sample_values[0]

                # move to left interval
                t1 = t0
                t0 = pp[i1 - 1]
                if t >= t0:
                    # we have arrived at the sought interval
                    self._cached_index = i1
                    return self._interpolate(i1, t0, t, t1)

            # prepare binary search on the left side of the index
            right = i1
            i1 = 0
        else:
            # the interval is cached, just interpolate
            self._cached_index = i1
            return self._interpolate(i1, t0, t, t1)

        # binary search
        while i1 < right:
            mid = (i1 + right) >> 1
            if t < pp[mid]:
                right = mid
            else:
                i1 = mid + 1

        # check boundary cases, again
        if i1 <= 0:
            self._cached_index = 0
            return self.sample_values[0]

        if i1 >= len_pp:
            i1 = len_pp
            self._cached_index = i1
            return self.sample_values[i1 - 1]

        t1 = pp[i1]
        t0 = pp[i1 - 1]

        self._cached_index = i1
        return self._interpolate(i1, t0, t, t1)

    def _interpolate(self, i1, t0, t, t1):
        raise NotImplementedError("call to abstract method")

    def __call__(self, t):
        return self.evaluate(t)


class LinearInterpolant(Interpolant):
    def __init__(self, parameter_positions, sample_values):
        super().__init__(parameter_positions, sample_values)

    def _interpolate(self, i1, t0, t, t1):
        values = self.sample_values

        alpha = (t - t0) / (t1 - t0)

        return values[i1 - 1] * (1 - alpha) + values[i1] * alpha


class QuaternionLinearInterpolant(Interpolant):
    """
    See: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#interpolation-slerp

    """

    def __init__(self, parameter_positions, sample_values):
        super().__init__(parameter_positions, sample_values)

    def _interpolate(self, i1, t0, t, t1):
        values = self.sample_values

        vk0 = values[i1 - 1]
        vk1 = values[i1]
        alpha = (t - t0) / (t1 - t0)

        dot = np.dot(vk0, vk1)

        if dot < 0:
            vk1 = -vk1
            dot = -dot

        if (
            dot > 0.99
        ):  # When theta is close to zero, spherical linear interpolation turns into regular linear interpolation.
            q = (1 - alpha) * vk0 + alpha * vk1
            return q / np.linalg.norm(q)  # remember to normalize the quaternion

        theta = np.arccos(dot)
        sin_theta = np.sin(theta)

        return (np.sin((1 - alpha) * theta) / sin_theta) * vk0 + (
            np.sin(alpha * theta) / sin_theta
        ) * vk1


class StepInterpolant(Interpolant):
    def __init__(self, parameter_positions, sample_values):
        super().__init__(parameter_positions, sample_values)

    def _interpolate(self, i1, t0, t, t1):
        return self.sample_values[i1 - 1]


class CubicSplineInterpolant(Interpolant):
    """
    See: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#interpolation-cubic
    Reference: https://github.com/mrdoob/three.js/blob/master/src/math/interpolants/CubicInterpolant.js
    """

    def __init__(self, parameter_positions, sample_values):
        super().__init__(parameter_positions, sample_values)

    def _interpolate(self, i1, t0, t, t1):
        pps = self.parameter_positions
        len_pps = len(pps)

        i_prev = i1 - 2
        i_next = i1 + 1

        if i_prev < 0:
            i_prev = i1
            t_prev = t1
        else:
            t_prev = pps[i_prev]

        if i_next >= len_pps:
            i_next = i1 - 1
            t_next = t0
        else:
            t_next = pps[i_next]

        half_dt = (t1 - t0) * 0.5

        w_p = half_dt / (t0 - t_prev)
        w_n = half_dt / (t_next - t1)

        p = (t - t0) / (t1 - t0)
        pp = p * p
        ppp = pp * p

        s_p = -w_p * ppp + 2 * w_p * pp - w_p * p
        s_0 = (1 + w_p) * ppp + (-1.5 - 2 * w_p) * pp + (-0.5 + w_p) * p + 1
        s_1 = (-1 - w_n) * ppp + (1.5 + w_n) * pp + 0.5 * p
        s_n = w_n * ppp - w_n * pp

        values = self.sample_values

        vk_prev = values[i_prev]
        vk_0 = values[i1 - 1]
        vk_1 = values[i1]
        vk_next = values[i_next]

        return s_p * vk_prev + s_0 * vk_0 + s_1 * vk_1 + s_n * vk_next
