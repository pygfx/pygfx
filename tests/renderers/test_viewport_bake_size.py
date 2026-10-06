"""
Test that bake functions are given the size of the viewport, not of the canvas.

``renderer.render(..., rect=...)`` draws into part of the canvas, and already
computes the logical size of that part in order to set the camera's view size.
Bake functions were handed the size of the whole canvas instead. The only bake
function in pygfx is the line shader's, which uses the size to convert ndc to
logical pixels, so a dashed line drawn into a sub-viewport came out with its
dashes scaled by the ratio between the canvas and the viewport -- about 2.6x too
fine in a typical side-by-side layout.

The test measures a property that does not depend on the adapter or on where the
line happens to land: the on-screen dash period of a line drawn into a viewport
must be the same as when the same line is drawn into the whole canvas.
"""

import numpy as np
import pytest
import wgpu

import pygfx as gfx

from ..testutils import can_use_wgpu_lib


if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


WIDTH, HEIGHT = 900, 400
# A viewport whose aspect differs strongly from the canvas, so that a wrong
# size is unmistakable rather than a rounding difference.
VIEW_WIDTH = 300
THICKNESS = 8.0
PATTERN = [2, 2]
# The pattern is measured in units of the thickness, so this is the period the
# dashes should have on screen, whatever they are drawn into.
EXPECTED_PERIOD = THICKNESS * sum(PATTERN)


def render_dashed_line(view_width=None, **material_kwargs):
    """A horizontal dashed line, drawn either full-canvas or into a viewport.

    The camera is always sized to the area being drawn into, so one model unit
    is one logical pixel either way and the dash period can be read straight off
    the image.
    """
    target = gfx.Texture(
        dim=2, size=(WIDTH, HEIGHT, 1), format=wgpu.TextureFormat.rgba8unorm
    )
    renderer = gfx.WgpuRenderer(target)
    renderer.ppaa = "none"
    renderer.pixel_ratio = 1

    positions = np.array([[-140, 0, 0], [140, 0, 0]], np.float32)
    scene = gfx.Scene()
    scene.add(gfx.Background.from_color("#000"))
    scene.add(
        gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineMaterial(
                thickness=THICKNESS,
                color="#fff",
                aa=False,
                dash_pattern=PATTERN,
                thickness_space="screen",
                **material_kwargs,
            ),
        )
    )

    width = view_width if view_width else WIDTH
    camera = gfx.OrthographicCamera(width, HEIGHT)
    if view_width:
        renderer.render(scene, camera, flush=False, rect=(0, 0, view_width, HEIGHT))
        renderer.flush()
    else:
        renderer.render(scene, camera)
    image = renderer.snapshot()[..., 0].astype(int)
    return image[:, :width]


def dash_period(image):
    """The mean on-screen dash period, along the line's own row."""
    ink = image > 128
    assert ink.any(), "nothing was drawn"
    row = ink[int(ink.sum(axis=1).argmax())]
    starts = int((np.diff(row.astype(int)) == 1).sum())
    assert starts >= 3, f"expected several dashes, found {starts}"
    lit = np.flatnonzero(row)
    return (lit.max() - lit.min()) / starts


def test_dash_period_is_the_same_in_a_viewport_as_full_canvas():
    """The dashes must not care which part of the canvas they are drawn into."""
    full = dash_period(render_dashed_line())
    viewport = dash_period(render_dashed_line(VIEW_WIDTH))

    assert full == pytest.approx(EXPECTED_PERIOD, rel=0.15), (
        f"the full-canvas reference is itself wrong: {full}"
    )
    assert viewport == pytest.approx(full, rel=0.1), (
        f"dashes differ between viewport ({viewport:.1f} px) and "
        f"full canvas ({full:.1f} px); the ratio of the two areas is "
        f"{WIDTH / VIEW_WIDTH:.2f}"
    )
