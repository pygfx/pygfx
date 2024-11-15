import numpy as np
from pygfx.animation.keyframe_track import KeyframeTrack


def test_keyframe_track_optimize():
    # (0,0,0,0,1,1,1,0,0,0,0,0,0,0) --> (0,0,1,1,0,0)
    times = np.array([0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    values = np.array([0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

    track = KeyframeTrack("test", None, None, times, values, lambda *_: None)
    assert np.all(track.times == np.array([0, 3, 4, 6, 7, 13]))
    assert np.all(track.values == np.array([0, 0, 1, 1, 0, 0]))

    values1 = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    track = KeyframeTrack("test", None, None, times, values1, lambda *_: None)
    assert np.all(track.times == np.array([0, 3, 4, 6, 7, 13]))
    assert np.all(
        track.values
        == np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]])
    )
