import time
import numpy as np
from pygfx import LinearInterpolant, CubicSplineInterpolant, StepInterpolant


def test_interpolant():
    times = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
    values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)

    linear_interpolant = LinearInterpolant(times, values)
    cubic_interpolant = CubicSplineInterpolant(times, values)
    step_interpolant = StepInterpolant(times, values)

    last_time = time.perf_counter()
    gloabl_time = 0

    for _ in range(100):
        now = time.perf_counter()
        dt = now - last_time
        last_time = now
        gloabl_time += dt

        scaled_gloabl_time = -gloabl_time  # test negative scale
        scaled_gloabl_time = scaled_gloabl_time % 0.9

        assert (
            linear_interpolant.evaluate(scaled_gloabl_time) - scaled_gloabl_time * 10
            < 1e-5
        )
        assert (
            cubic_interpolant.evaluate(scaled_gloabl_time) - scaled_gloabl_time * 10
            < 1e-5
        )
        assert step_interpolant.evaluate(scaled_gloabl_time) == np.floor(
            scaled_gloabl_time * 10
        )

        random_sleep = np.random.rand() * 0.05
        time.sleep(random_sleep)
