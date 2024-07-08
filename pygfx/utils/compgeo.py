"""Computational geometry.

This may at some pointe be moved to https://github.com/pygfx/pycompgeo.
"""

import bisect


def get_visible_part_of_line_ndc(ndc1, ndc2):
    """Get the visible part of the line, given by two homogeneous ndc coords.

    Returns (t1, t2) representing the visible line. If the two values
    are equal, the line is not in the viewport.
    """

    # Uncomment for performance measurements
    import time

    time_0 = time.perf_counter()

    # Get closest t for all 4 edges
    tx1 = binary_search_for_ndc_edge(ndc1, ndc2, -1, 0)
    tx2 = binary_search_for_ndc_edge(ndc1, ndc2, +1, 0)
    ty1 = binary_search_for_ndc_edge(ndc1, ndc2, -1, 1)
    ty2 = binary_search_for_ndc_edge(ndc1, ndc2, +1, 1)

    # Sort (because line can be oriented in any way)
    tx1, tx2 = min(tx1, tx2), max(tx1, tx2)
    ty1, ty2 = min(ty1, ty2), max(ty1, ty2)

    # Combine dimensions
    t1, t2 = max(tx1, ty1), min(tx2, ty2)

    # Sort again
    t1, t2 = min(t1, t2), max(t1, t2)

    print(f"{time.perf_counter() - time_0:f}")

    return t1, t2


def binary_search_for_ndc_edge(ndc1, ndc2, ref, dim, *, n_iters=10):
    """Get the t that results in the smallest distance to ref in dimension dim.

    The ndc1 and ndc2 arguments must be in homogeneous ndc coords, so
    that they can be interpolated. The resulting t represents the point
    on the line between ndc1 and ncd2.

    This algorithm operates either over x or y, depending on dim (0 or1).
    So in order to determine the visible piece of the line, this
    function must be called 4 times.
    """

    # This algorithm performs a binary search, so that we can cover a
    # fine grid without doing an exhaustive search.
    #
    # Note that (with a perspective camera) the distance to the
    # reference does not scale linearly with t. The consequence is that
    # there is not a closed form solution to this problem. However, for
    # an orthographic camera there is!
    #
    # The binary search is done by sampling three points, then determining
    # whether the ref value is in 0..1 or 1..2, and then sampling one new
    # value in between. Repeat.
    #
    #  |------|------|
    #  0      1      2
    #
    # AK: I spent some time optimizing this code. Using numpy arrays
    # and np.argmin is not faster. What does help is re-using the sample
    # positions. With 5 samples per level, we can re-use 2 or 3 of the
    # samples at each step. By using tuples, copying these into the new
    # slot is efficient.

    # Reduce the ndc positions to two scalars per position
    x1, x2 = float(ndc1[dim]), float(ndc2[dim])
    w1, w2 = float(ndc1[3]), float(ndc2[3])

    # Get the starting t's.
    # Things get iffy when the w is zero or negative, so we first find
    # the range where w is positive.
    initial_t1, initial_t2 = 0.0, 1.0
    eps = 1e-8
    if w1 < eps or w2 < eps:
        if w1 < eps and w2 < eps:
            # Camera is perspective and the end-points are both behind the camera.
            return 0.0 if w1 >= w2 else 1.0
        elif w1 < eps:
            # The first point is behind, move it.
            initial_t1 = (eps - w1) / (w2 - w1)
        else:  # w2 < eps:
            # The second point is behind, move it.
            initial_t2 = 1.0 - (eps - w2) / (w1 - w2)

    def new_sample(t):
        x = x1 * (1 - t) + x2 * t
        w = w1 * (1 - t) + w2 * t
        return t, x / w

    # Produce 3 tuples (t, value) representing the relative t-values as shown in above ascii diagram
    samples = [
        new_sample(initial_t1),
        new_sample(0.5 * (initial_t1 + initial_t2)),
        new_sample(initial_t2),
    ]

    # If the values are very close, the line could be a point, or orthogonal to this dim
    if abs(samples[0][1] - samples[2][1]) < 1e-9:
        return 1.0 if samples[0][1] < ref else 0.0

    # Determine function for bisect. Bisect needs a sorted list, this handles
    # the case where the order is reversed.
    key_func_forward = lambda sample: sample[1] - ref
    key_func_reverse = lambda sample: ref - sample[1]
    if samples[0][1] <= samples[2][1]:
        key_func = key_func_forward
    else:
        key_func = key_func_reverse

    # Do the first bisection!
    i = bisect.bisect(samples, 0.0, key=key_func)

    # Go deeper, if we must
    if w1 == 1.0 and w2 == 1.0:
        pass  # Ortho camera
    elif i == 0 or i == 3:
        pass  # The whole line is on one side of the ref
    else:
        # Binary search. With 10 iters we cover over 1K samples.
        for _ in range(n_iters):
            if i <= 1:
                samples[2] = samples[1]
            else:
                samples[0] = samples[1]
            samples[1] = new_sample(0.5 * (samples[0][0] + samples[2][0]))
            i = bisect.bisect(samples, 0.0, key=key_func)

    # Fine tune using a linear fit
    t_step = samples[1][0] - samples[0][0]
    if i == 0:
        t = samples[0][0]
    elif i == 3:
        t = samples[2][0]
    elif i == 1:
        y1 = samples[0][1]
        y2 = samples[1][1]
        dt = (ref - y1) / (y2 - y1)
        t = samples[0][0] + dt * t_step
    else:  # i == 2
        y1 = samples[1][1]
        y2 = samples[2][1]
        dt = (ref - y1) / (y2 - y1)
        t = samples[1][0] + dt * t_step

    return t
