import numpy as np


class KeyframeTrack:
    def __init__(self, name, target, path, times, values, interpolation):
        self.name = name
        self.target = target
        self.path = path

        assert len(times) == len(values), "times and values must have the same length"

        self._optimize(times, values)
        self.interpolation = interpolation(times, values)

    def _optimize(self, times, values):
        # removes equivalent sequential keys as common in morph target sequences
        # (0,0,0,0,1,1,1,0,0,0,0,0,0,0) --> (0,0,1,1,0,0)
        optimized_times = []
        optimized_values = []
        prev_time = None
        prev_value = None
        for time, value in zip(times, values, strict=True):
            # remove adjacent keyframes scheduled at the same times
            if time != prev_time:
                if np.any(value != prev_value):
                    if prev_time is not None:
                        optimized_times.append(prev_time)
                        optimized_values.append(prev_value)
                    optimized_times.append(time)
                    optimized_values.append(value)
                    prev_value = value

                prev_time = time

        # ensure the last keyframe is added
        if optimized_times[-1] != times[-1]:
            optimized_times.append(times[-1])
            optimized_values.append(values[-1])

        self.times = np.array(optimized_times)
        self.values = np.array(optimized_values)
        return self.times, self.values
