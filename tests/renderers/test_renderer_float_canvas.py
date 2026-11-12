"""Canvas output encoding and floating-point bitmap downloads."""

import subprocess
import sys

import numpy as np
import pytest
import wgpu
from rendercanvas.offscreen import RenderCanvas

import pygfx as gfx

from ..testutils import can_use_wgpu_lib


if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


class RecordingCanvas(RenderCanvas):
    def _rc_present_bitmap(self, *, data, format, **kwargs):
        self.last_bitmap_format = format
        super()._rc_present_bitmap(data=data, format=format, **kwargs)


def _check_canvas_output(bitmap_format, gamma, *, force_unorm=False):
    canvas = RecordingCanvas(size=(17, 5), format=bitmap_format)
    if force_unorm:
        # Exercise the existing fallback for a canvas without an sRGB format.
        canvas.get_context("wgpu").get_preferred_format = lambda adapter: "rgba8unorm"
    renderer = gfx.WgpuRenderer(
        canvas, pixel_scale=1, ppaa="none", gamma_correction=gamma
    )
    linear = np.array((0.25, 0.1, 0.03))
    color = gfx.Color.from_physical(*linear)
    scene = gfx.Scene().add(gfx.Background(None, gfx.BackgroundMaterial(color)))
    canvas.request_draw(lambda: renderer.render(scene, gfx.NDCCamera()))

    is_float = bitmap_format in ("rgba-f16", "rgba-f32")
    if is_float:
        dtype = np.float16 if bitmap_format == "rgba-f16" else np.float32
        expected_rgb = linear**gamma
    else:
        dtype = np.uint8
        if force_unorm:
            expected_rgb = linear ** (gamma / 2.2)
        else:
            expected_rgb = gfx.Color.from_physical(*(linear**gamma))[:3]
    expected = np.array([*expected_rgb, 1.0])

    # Both sizes require padded GPU rows for float16 and float32. Check that
    # resizing and downloading remove padding without changing dtype or format.
    for width, height in ((17, 5), (33, 7)):
        canvas.set_logical_size(width, height)
        image = canvas.draw()
        assert canvas.last_bitmap_format == bitmap_format
        assert image.dtype == dtype
        assert image.shape == (height, width, 4)
        assert image.flags.c_contiguous
        values = image if is_float else image / 255
        np.testing.assert_allclose(
            values, np.broadcast_to(expected, values.shape), atol=0.004
        )


@pytest.mark.parametrize("gamma", [1.0, 0.5])
def test_float16_canvas_output(gamma):
    _check_canvas_output("rgba-f16", gamma)


@pytest.mark.parametrize("gamma", [1.0, 0.5])
def test_float32_canvas_output(gamma):
    required = {"float32-blendable", "float32-filterable"}
    if not required.issubset(wgpu.gpu.request_adapter_sync().features):
        pytest.skip("Float32 canvas output needs float32 blending and filtering")

    # Enable optional features before creating the shared device, independently
    # of other tests that use a device configured with the default features.
    code = f"""
import pygfx as gfx
gfx.renderers.wgpu.enable_wgpu_features('float32-blendable', 'float32-filterable')
from tests.renderers.test_renderer_float_canvas import _check_canvas_output
_check_canvas_output('rgba-f32', {gamma!r})
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("force_unorm", [False, True])
def test_uint8_canvas_output(force_unorm):
    _check_canvas_output("rgba-u8", 1.0, force_unorm=force_unorm)
