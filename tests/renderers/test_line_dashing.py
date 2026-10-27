"""
Test the cumulative distance that the line shader bakes to parametrize dashes.

The buffer is checked directly (rather than via a screenshot) because the
values have exact expected answers, which makes the assertions sharp.
"""

import numpy as np
import pygfx as gfx
from pygfx.renderers.wgpu.shaders.lineshader import LineShader


NAN = np.full((1, 3), np.nan, np.float32)


def regular_polygon(n, x=0.0, r=1.0):
    """The nodes of a regular n-gon, without repeating the first one."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack(
        [x + r * np.sin(t), r * np.cos(t), np.zeros_like(t)], axis=1
    ).astype(np.float32)


def polygon_side_length(n, r):
    return 2 * r * np.sin(np.pi / n)


def bake(positions, **material_kwargs):
    """Bake the line distance buffer for the given positions, and return it."""
    material_kwargs.setdefault("thickness_space", "model")
    line = gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineMaterial(thickness=1, dash_pattern=[2, 2], **material_kwargs),
    )
    camera = gfx.OrthographicCamera(100, 100)
    shader = LineShader(line)
    shader.bake_function(line, camera, (100, 100))
    return shader.line_distance_buffer.data


def test_cumdist_of_a_plain_line():
    positions = np.array([[0, 0, 0], [3, 0, 0], [3, 4, 0]], np.float32)
    cumdist = bake(positions)
    assert np.allclose(cumdist, [0, 3, 7])


def test_cumdist_restarts_at_each_line_piece():
    """Each nan-separated piece starts its dash pattern from scratch.

    Without this, the dash phase of a piece is offset by the total length of
    all the pieces before it, so the dashes of the later pieces race whenever
    something changes those lengths (such as the zoom level).
    """
    positions = np.vstack(
        [
            np.array([[0, 0, 0], [3, 0, 0], [3, 4, 0]], np.float32),
            NAN,
            np.array([[0, 0, 0], [5, 0, 0]], np.float32),
            NAN,
            NAN,  # successive nans must not confuse the piece detection
            np.array([[0, 0, 0], [0, 2, 0]], np.float32),
        ]
    ).astype(np.float32)
    cumdist = bake(positions)
    assert np.allclose(cumdist[0:3], [0, 3, 7])
    assert np.allclose(cumdist[4:6], [0, 5])
    assert np.allclose(cumdist[8:10], [0, 2])


def test_cumdist_closes_a_loop():
    """The node that closes a loop holds the cumdist of the *whole* loop.

    If it held the cumdist of the loop's first node instead, the shader would
    measure the closing segment as if it spanned the entire loop, and its
    dashes would come out that many times too dense. See gh-1103.
    """
    n, r = 5, 10.0
    side = polygon_side_length(n, r)
    cumdist = bake(regular_polygon(n, r=r), loop=True)

    # One extra element, to hold the cumdist of the closed loop
    assert len(cumdist) == n + 1
    assert np.allclose(cumdist, side * np.arange(n + 1))


def test_cumdist_closes_several_loops():
    """Every loop closes on itself, and each one starts from zero."""
    shapes = [(4, 10.0), (3, 5.0), (6, 7.0)]
    positions = np.vstack(
        [
            x
            for n, r in shapes
            for x in (regular_polygon(n, r=r), NAN)
            # the last nan is harmless: a trailing nan is not a loop
        ]
    ).astype(np.float32)
    cumdist = bake(positions, loop=True)

    i = 0
    for n, r in shapes:
        side = polygon_side_length(n, r)
        assert np.allclose(cumdist[i : i + n + 1], side * np.arange(n + 1))
        i += n + 1


def test_cumdist_honors_the_draw_range():
    """Loops are detected relative to the draw range, at the right nodes."""
    n, r = 4, 10.0
    side = polygon_side_length(n, r)
    positions = np.vstack([NAN, NAN, regular_polygon(n, r=r), NAN]).astype(np.float32)
    line = gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineMaterial(
            thickness=1, dash_pattern=[2, 2], loop=True, thickness_space="model"
        ),
    )
    line.geometry.positions.draw_range = 2, n + 1
    camera = gfx.OrthographicCamera(100, 100)
    shader = LineShader(line)
    shader.bake_function(line, camera, (100, 100))

    cumdist = shader.line_distance_buffer.data
    assert np.allclose(cumdist[2 : 2 + n + 1], side * np.arange(n + 1))


def test_cumdist_with_nonfinites_in_other_thickness_spaces():
    """Nans must not trip up the transform to world/screen space."""
    positions = np.vstack([regular_polygon(4, r=10.0), NAN]).astype(np.float32)
    for thickness_space in ["model", "world", "screen"]:
        cumdist = bake(positions, loop=True, thickness_space=thickness_space)
        assert np.all(np.isfinite(cumdist))
        assert cumdist[0] == 0
        assert cumdist[4] > cumdist[3] > 0


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            func()
