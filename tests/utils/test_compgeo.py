from pygfx.utils.compgeo import binary_search_for_ndc_edge, bisect_asc, bisect_desc


import bisect


def test_bisect():
    values = [0, 1, 2, 3, 3.2, 4, 5]

    for ref in [-9, -1, 0, 1, 2, 3, 4, 5, 6, 9, 2.1, 2.5, 2.9, 3.1, 3.2, 3.3]:
        i0 = bisect.bisect(values, ref)
        i1 = bisect_asc(values, ref)
        i2 = bisect_desc(reversed(values), ref)
        assert i0 == i1
        assert i0 == len(values) - i2


def test_binary_search_for_ndc_edge_fit():
    for ndc1, ndc2, ref, real_t in [
        #
        ((4, 0, 0, 1), (7, 0, 0, 1), 5, 1 / 3),
        ((4, 0, 0, 1), (7, 0, 0, 1), 5.5, 0.5),
        ((4, 0, 0, 1), (7, 0, 0, 1), 6, 2 / 3),
        #
        ((7, 0, 0, 1), (4, 0, 0, 1), 5, 2 / 3),
        ((7, 0, 0, 1), (4, 0, 0, 1), 5.5, 0.5),
        ((7, 0, 0, 1), (4, 0, 0, 1), 6, 1 / 3),
        #
        ((-100, 0, 0, 1), (-110, 0, 0, 1), -102, 2 / 10),
        ((-100, 0, 0, 1), (-110, 0, 0, 1), -106, 6 / 10),
        ((-100, 0, 0, 1), (-110, 0, 0, 1), -107, 7 / 10),
        #
        ((-110, 0, 0, 1), (-100, 0, 0, 1), -102, 8 / 10),
        ((-110, 0, 0, 1), (-100, 0, 0, 1), -106, 4 / 10),
        ((-110, 0, 0, 1), (-100, 0, 0, 1), -107, 3 / 10),
    ]:
        # With only a quadratic fit
        t = binary_search_for_ndc_edge(ndc1, ndc2, ref, 0, n_iters=0)
        fault = abs(real_t - t)
        assert fault < 1e-6

        # Result should get better as iters is increased
        last_t = t
        for iters in range(1, 12):
            t = binary_search_for_ndc_edge(ndc1, ndc2, ref, 0, n_iters=iters)
            fault = abs(real_t - t)
            if fault > 1e-15:
                last_fault = abs(real_t - last_t)
                assert fault < last_fault
            last_t = t

        assert fault < 3e-5


def test_binary_search_for_ndc_edge_perspective():
    ndc1 = (1, 0, 0, 1)  # 1
    ndc2 = (2, 0, 0, 1)  # 2
    t = binary_search_for_ndc_edge(ndc1, ndc2, 1.5, 0, n_iters=10)
    assert t == 0.5

    ndc1 = (2, 0, 0, 2)  # 1
    ndc2 = (8, 0, 0, 4)  # 2
    t = binary_search_for_ndc_edge(ndc1, ndc2, 1.5, 0, n_iters=10)
    assert abs(t - 1 / 3) < 0.0001

    ndc1 = (3, 0, 0, 3)  # 1
    ndc2 = (10, 0, 0, 5)  # 2
    t = binary_search_for_ndc_edge(ndc1, ndc2, 1.5, 0, n_iters=10)
    assert abs(t - 0.375) < 0.0001


def test_binary_search_for_ndc_edge_ortho():
    # This line is on-screen, also in y dimension!
    ndc1 = (0.1, 0, 0, 1)  # 1
    ndc2 = (0.8, 0, 0, 1)  # 2
    t1 = binary_search_for_ndc_edge(ndc1, ndc2, -1, 1, n_iters=10)
    t2 = binary_search_for_ndc_edge(ndc1, ndc2, +1, 1, n_iters=10)
    assert t1 == 0.0 and t2 == 1.0

    # This line is above viewport
    ndc1 = (0.1, -1.2, 0, 1)  # 1
    ndc2 = (0.8, -1.2, 0, 1)  # 2
    t1 = binary_search_for_ndc_edge(ndc1, ndc2, -1, 1, n_iters=10)
    t2 = binary_search_for_ndc_edge(ndc1, ndc2, +1, 1, n_iters=10)
    assert t1 == 1.0 and t2 == 1.0

    # This line is below viewport
    ndc1 = (0.1, 1.2, 0, 1)  # 1
    ndc2 = (0.8, 1.2, 0, 1)  # 2
    t1 = binary_search_for_ndc_edge(ndc1, ndc2, -1, 1, n_iters=10)
    t2 = binary_search_for_ndc_edge(ndc1, ndc2, +1, 1, n_iters=10)
    assert t1 == 0.0 and t2 == 0.0


if __name__ == "__main__":
    test_bisect()
    test_binary_search_for_ndc_edge_fit()
    test_binary_search_for_ndc_edge_ortho()
    test_binary_search_for_ndc_edge_perspective()
