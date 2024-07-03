"""Computational geometry
"""


def get_visible_part_of_line_ndc(ndc1, ndc2):
    """Get the visible part of the line, given by two homogeneous ndc coords."""

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
    # reference does not scale linear with t. This means that if we
    # sample two points, we cannot simply select the half that has the
    # smallest distance, since the minimum can still be in the other
    # half. To work around this, we sample 5 locations, and select a
    # sub-region based on the point with smallest distance. When that
    # point is on the edge (0 or 4), the subregion is just 25% of the
    # previous. Otherwise (1, 2, 3) it is 50%.
    #
    #  |------|------|------|------|
    #  0      1      2      3      4
    #
    # AK: I spent some time optimizing this code. Using numpy arrays
    # and np.argmin is not faster. What does help is re-using the sample
    # positions. With 5 samples per level, we can re-use 2 or 3 of the
    # samples at each step. By using tuples, copying these into the new
    # slot is efficient.

    # Reduce the ndc positions to two scalars per position
    x1, x2 = ndc1[dim], ndc2[dim]
    w1, w2 = ndc1[3], ndc2[3]

    def new_sample(t):
        x = x1 * (1 - t) + x2 * t
        w = w1 * (1 - t) + w2 * t
        w = max(1e-9, w)  # sometimes w is negative, resulting in wrong results
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

    # Binary search. Wth 10 iters we cover over 1K samples.
    for _ in range(n_iters):
        i = argmin(samples)

        # To emulate a binary search assuming a linear relation between t and dist
        # i = 1 if samples[0][1] < samples[4][1] else 3  # WRONG!

        if i == 0:
            # samples[0] = samples[0]
            samples[4] = samples[1]
            samples[2] = new_sample(0.5 * (samples[0][0] + samples[4][0]))
        elif i == 4:
            samples[0] = samples[3]
            # samples[4] = samples[4]
            samples[2] = new_sample(0.5 * (samples[0][0] + samples[4][0]))
        elif i == 1:
            # samples[0] = samples[0]
            samples[4] = samples[2]
            samples[2] = samples[1]
        elif i == 3:
            samples[0] = samples[2]
            # samples[4] = samples[4]
            samples[2] = samples[3]
        else:  # i == 2:
            samples[0] = samples[1]
            samples[4] = samples[3]
            # samples[2] = samples[2]
        samples[1] = new_sample(0.5 * (samples[0][0] + samples[2][0]))
        samples[3] = new_sample(0.5 * (samples[2][0] + samples[4][0]))

    # Select the sample that is closest.
    i = argmin(samples)
    t, dist = samples[i]
    t_step = samples[1][0] - samples[0][0]

    # If this is on (or close to) the edge, this is it
    if t < t_step or t > 1.0 - t_step:
        return t

    # Otherwise, perform a quadratic fit find the sub-sample result.
    # This optimization means we need less depth in the binary search to still
    # found an accurate solution.
    y1 = samples[i - 1][1] if i > 0 else new_sample(t - t_step)[1]
    y2 = dist
    y3 = samples[i + 1][1] if i < 5 else new_sample(t + t_step)[1]
    num = (y3 - y2) - (y1 - y2)
    denom = 2 * (y1 + y3 - 2 * y2)
    t_fine = 0.0
    if denom != 0:
        t_fine = -num / denom
    return t + t_fine * t_step
