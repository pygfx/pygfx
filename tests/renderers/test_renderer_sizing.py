import pytest
import pygfx as gfx

from rendercanvas.offscreen import RenderCanvas

from ..testutils import can_use_wgpu_lib


# The renderer will create a wgpu context, so need wgpu, even though we don't render anything
if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


def test_renderer_sizing_auto_hires():
    # Create canvas with pixel ratio 1
    canvas = RenderCanvas(size=(100, 100), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)

    assert renderer.pixel_scale == 2
    assert renderer.pixel_ratio == 2
    assert renderer.physical_size == (200, 200)

    # Create canvas with pixel ratio 2
    canvas = RenderCanvas(size=(100, 100), pixel_ratio=2)
    renderer = gfx.WgpuRenderer(canvas)

    assert renderer.pixel_scale == 1
    assert renderer.pixel_ratio == 2
    assert renderer.physical_size == (200, 200)

    # Change pixel ratio, does not change physical size
    canvas.set_pixel_ratio(3)

    assert renderer.pixel_scale == 1
    assert renderer.pixel_ratio == 3
    assert renderer.physical_size == (200, 200)

    # Change pixel ratio, does not change physical size.
    # But since pygfx now enters ssaa mode, its physical size does change
    canvas.set_pixel_ratio(1)

    assert renderer.pixel_scale == 2
    assert renderer.pixel_ratio == 2
    assert renderer.physical_size == (400, 400)


def test_renderer_sizing_set_pixel_scale():
    # Create canvas with pixel ratio 1
    canvas = RenderCanvas(size=(100, 100), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas, pixel_scale=1)

    assert renderer.pixel_scale == 1
    assert renderer.pixel_ratio == 1
    assert renderer.physical_size == (100, 100)

    # Create canvas with pixel ratio 2
    canvas = RenderCanvas(size=(100, 100), pixel_ratio=2)
    renderer = gfx.WgpuRenderer(canvas, pixel_scale=1)

    assert renderer.pixel_scale == 1
    assert renderer.pixel_ratio == 2
    assert renderer.physical_size == (200, 200)

    # Change pixel ratio, does not change physical size
    canvas.set_pixel_ratio(3)

    assert renderer.pixel_scale == 1
    assert renderer.pixel_ratio == 3
    assert renderer.physical_size == (200, 200)

    # Change pixel ratio, does not change physical size.
    # Since the renderer as fixed pixel-scale, it does not enter ssaa mode
    canvas.set_pixel_ratio(1)

    assert renderer.pixel_scale == 1
    assert renderer.pixel_ratio == 1
    assert renderer.physical_size == (200, 200)


def test_renderer_sizing_set_pixel_ratio():
    # Create canvas with pixel ratio 1
    canvas = RenderCanvas(size=(100, 100), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas, pixel_ratio=1)

    assert renderer.pixel_scale == 1
    assert renderer.pixel_ratio == 1
    assert renderer.physical_size == (100, 100)

    # Create canvas with pixel ratio 2
    canvas = RenderCanvas(size=(100, 100), pixel_ratio=2)
    renderer = gfx.WgpuRenderer(canvas, pixel_ratio=1)

    assert renderer.pixel_scale == 0.5
    assert renderer.pixel_ratio == 1
    assert renderer.physical_size == (100, 100)

    # Change pixel ratio, does not change physical size
    canvas.set_pixel_ratio(3)

    assert renderer.pixel_scale == 1 / 3
    assert renderer.pixel_ratio == 1
    assert renderer.physical_size == (66, 66)

    # Change pixel ratio, does not change physical size.
    # Since the renderer as fixed pixel-ratio, it does not enter ssaa mode
    canvas.set_pixel_ratio(1)

    assert renderer.pixel_scale == 1
    assert renderer.pixel_ratio == 1
    assert renderer.physical_size == (200, 200)
