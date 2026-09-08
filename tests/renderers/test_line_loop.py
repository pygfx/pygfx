"""Test looping lines, in particular the fixed-size loops of ``material.loop = n``."""

import numpy as np
import pygfx as gfx
from pygfx.renderers.wgpu.shaders.lineshader import LineShader


def make_shader(n_nodes, loop, draw_range=None):
    line = gfx.Line(
        gfx.Geometry(positions=np.zeros((n_nodes, 3), np.float32)),
        gfx.LineMaterial(loop=loop),
    )
    if draw_range is not None:
        line.geometry.positions.draw_range = draw_range
    return line, LineShader(line)


def test_material_loop_accepts_bools_and_ints():
    assert gfx.LineMaterial(loop=False).loop is False
    assert gfx.LineMaterial(loop=True).loop is True
    assert gfx.LineMaterial(loop=0).loop is False
    assert gfx.LineMaterial(loop=4).loop == 4
    assert gfx.LineMaterial(loop=np.int64(4)).loop == 4

    # A loop needs at least 3 nodes, so values below that cannot denote a shape
    # size, and keep their original boolean meaning.
    for value in [1, 2, -3]:
        assert gfx.LineMaterial(loop=value).loop is True


def test_fixed_size_loops_need_no_extra_buffers():
    """The whole point: no per-node buffer, and no baking, just index math."""
    _, shader = make_shader(12, loop=4)
    assert shader["loop"] is True
    assert shader["loop_size"] == 4
    assert not hasattr(shader, "line_loop_buffer")
    assert not shader.needs_bake_function

    # Whereas nan-separated loops do need a buffer
    _, shader = make_shader(12, loop=True)
    assert shader["loop_size"] == 0
    assert hasattr(shader, "line_loop_buffer")


def test_fixed_size_loops_draw_one_extra_node_per_shape():
    """Each shape of n nodes is drawn as n + 1 (virtual) nodes, 6 vertices each."""
    line, shader = make_shader(12, loop=4)
    offset, size = shader._get_n(line.geometry.positions)
    assert (offset, size) == (0, 3 * 5 * 6)


def test_fixed_size_loops_ignore_an_incomplete_trailing_shape():
    line, shader = make_shader(14, loop=4)  # 3 shapes and 2 spare nodes
    offset, size = shader._get_n(line.geometry.positions)
    assert (offset, size) == (0, 3 * 5 * 6)


def test_fixed_size_loops_honor_the_draw_range():
    # The 2nd and 3rd shape
    line, shader = make_shader(20, loop=4, draw_range=(4, 8))
    offset, size = shader._get_n(line.geometry.positions)
    assert (offset, size) == (1 * 5 * 6, 2 * 5 * 6)

    # A range that does not start on a shape boundary snaps back to one
    line, shader = make_shader(20, loop=4, draw_range=(2, 8))
    offset, size = shader._get_n(line.geometry.positions)
    assert (offset, size) == (0, 2 * 5 * 6)

    # A range that is too small for a single shape draws nothing
    line, shader = make_shader(20, loop=4, draw_range=(4, 3))
    offset, size = shader._get_n(line.geometry.positions)
    assert size == 0


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            func()
