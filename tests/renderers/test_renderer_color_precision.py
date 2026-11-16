"""Precision of internal color buffers and their readback paths."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import wgpu

import pygfx as gfx
from pygfx.renderers.wgpu.engine.blender import Blender
from pygfx.renderers.wgpu.engine.effectpasses import CopyPass
from rendercanvas.offscreen import RenderCanvas

from ..testutils import can_use_wgpu_lib


if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


@pytest.mark.parametrize("precision", [None, "uint16", "FLOAT16", 16, []])
def test_invalid_color_precision(precision):
    target = gfx.Texture(dim=2, size=(16, 16, 1), format="rgba8unorm")
    with pytest.raises(ValueError, match="color_precision"):
        gfx.WgpuRenderer(target, color_precision=precision)
    with pytest.raises(ValueError, match="color_precision"):
        Blender(color_precision=precision)


def test_color_precision_targets():
    default = Blender()
    floating = Blender(color_precision="float16")
    another_default = Blender()
    assert default.hash == another_default.hash
    assert default.hash != floating.hash
    for name, info in default.texture_info.items():
        if name in ("color", "altcolor"):
            assert info["format"] == "rgba8unorm-srgb"
            assert floating.texture_info[name]["format"] == "rgba16float"
        else:
            assert floating.texture_info[name] == info


@pytest.mark.parametrize("ppaa", ["none", "fxaa", "ddaa"])
@pytest.mark.parametrize("precision", ["uint8", "float16"])
def test_color_precision_render(precision, ppaa):
    dtype = np.float32 if precision == "float32" else np.float16
    tolerance = 0.00001 if precision == "float32" else 0.002
    target = gfx.Texture(
        dim=2,
        size=(16, 16, 1),
        format="rgba32float" if precision == "float32" else "rgba16float",
    )
    target._wgpu_usage |= wgpu.TextureUsage.COPY_SRC
    renderer = gfx.WgpuRenderer(
        target, color_precision=precision, pixel_scale=1, ppaa=ppaa
    )
    # Exercise both ping-pong buffers, also when PPAA is disabled.
    renderer.effect_passes = (CopyPass(), *renderer.effect_passes)
    # These values distinguish float32 from float16 rounding as well as uint8 clipping.
    expected = (
        (70000.125, 2.0003, 0.125007, 1.0)
        if precision == "float32"
        else (2.0003, 0.25003, 0.125007, 1.0)
    )
    color = gfx.Color.from_physical(*expected)
    mesh = gfx.Mesh(
        gfx.plane_geometry(4, 4), gfx.MeshBasicMaterial(color=color, pick_write=True)
    )
    scene = gfx.Scene().add(mesh)
    camera = gfx.OrthographicCamera(2, 2)

    renderer.render(scene, camera)
    assert renderer.color_precision == precision
    image = renderer.snapshot()
    assert image.shape == (16, 16, 4)
    if precision != "uint8":
        assert image.dtype == dtype
        np.testing.assert_allclose(image[8, 8], expected, atol=tolerance, rtol=0.000002)
    else:
        assert image.dtype == np.uint8
        # The default buffer stores sRGB and clips HDR to 1.
        expected_srgb = np.array([*color[:3], 1.0]).clip(0, 1)
        np.testing.assert_allclose(image[8, 8] / 255, expected_srgb, atol=1 / 255)

    picked = renderer.get_pick_info((8, 8))
    assert picked["world_object"] is mesh
    if precision != "uint8":
        np.testing.assert_allclose(picked["rgba"], color, atol=tolerance, rtol=0.000002)
        # The final copy to a float texture must preserve linear HDR values too.
        data = renderer.device.queue.read_texture(
            {"texture": target._wgpu_object},
            {"bytes_per_row": 16 * 4 * np.dtype(dtype).itemsize},
            (16, 16, 1),
        )
        output = np.frombuffer(data, dtype).reshape(16, 16, 4)
        np.testing.assert_allclose(
            output[8, 8], expected, atol=tolerance, rtol=0.000002
        )
    else:
        np.testing.assert_allclose(picked["rgba"], expected_srgb, atol=1 / 255)


def test_default_color_precision():
    canvas = RenderCanvas(size=(16, 16))
    default = gfx.WgpuRenderer(canvas, pixel_scale=1)
    explicit = gfx.WgpuRenderer(canvas, color_precision="uint8", pixel_scale=1)
    scene = gfx.Scene().add(gfx.Background(None, gfx.BackgroundMaterial("#2d7ab1")))
    for renderer in (default, explicit):
        renderer.render(scene, gfx.NDCCamera())
    assert default.color_precision == "uint8"
    np.testing.assert_array_equal(default.snapshot(), explicit.snapshot())


@pytest.mark.parametrize("alpha_mode", ["blend", "weighted_blend"])
def test_color_precision_shared_scene(alpha_mode, floating_precision="float16"):
    # The same object needs separate pipelines for the two color formats.
    color = gfx.Color.from_physical(4.0, 0.5, 0.25, 0.5)
    mesh = gfx.Mesh(
        gfx.plane_geometry(4, 4),
        gfx.MeshBasicMaterial(color=color, alpha_mode=alpha_mode),
    )
    scene = gfx.Scene().add(mesh)
    camera = gfx.OrthographicCamera(2, 2)
    renderers = [
        gfx.WgpuRenderer(
            gfx.Texture(dim=2, size=(16, 16, 1), format="rgba8unorm"),
            color_precision=precision,
            pixel_scale=1,
            ppaa="none",
        )
        for precision in ("uint8", floating_precision)
    ]
    for _ in range(2):
        for renderer in renderers:
            renderer.render(scene, camera)
        floating = renderers[1].snapshot()[8, 8]
        # Both blending paths store premultiplied color. Revealage is uint8,
        # so weighted blending introduces a small additional rounding error.
        np.testing.assert_allclose(floating, (2.0, 0.25, 0.125, 0.5), atol=0.01)


@pytest.mark.parametrize("missing", ["float32-blendable", "float32-filterable"])
def test_float32_requires_features(monkeypatch, missing):
    from pygfx.renderers.wgpu.engine import blender

    features = {"float32-blendable", "float32-filterable"} - {missing}
    shared = SimpleNamespace(device=SimpleNamespace(features=features))
    monkeypatch.setattr(blender, "get_shared", lambda: shared)
    target = gfx.Texture(dim=2, size=(16, 16, 1), format="rgba8unorm")
    with pytest.raises(RuntimeError, match="enable_wgpu_features"):
        gfx.WgpuRenderer(target, color_precision="float32")
    with pytest.raises(RuntimeError, match="enable_wgpu_features"):
        Blender(color_precision="float32")


@pytest.mark.parametrize("ppaa", ["none", "fxaa", "ddaa"])
def test_float32_rendering(ppaa):
    required = {"float32-blendable", "float32-filterable"}
    if not required.issubset(wgpu.gpu.request_adapter_sync().features):
        pytest.skip(
            "Float32 rendering requires float32-blendable and float32-filterable"
        )

    # Feature selection must precede the first renderer. Isolate it from tests
    # that have already created the shared device with the default features.
    code = f"""
import pygfx as gfx
gfx.renderers.wgpu.enable_wgpu_features('float32-blendable', 'float32-filterable')
from tests.renderers.test_renderer_color_precision import (
    test_color_precision_render, test_color_precision_shared_scene,
)
from pygfx.renderers.wgpu.engine.blender import Blender
floating = Blender(color_precision='float32')
assert floating.texture_info['color']['format'] == 'rgba32float'
assert floating.texture_info['altcolor']['format'] == 'rgba32float'
assert floating.hash != Blender().hash
test_color_precision_render('float32', {ppaa!r})
for alpha_mode in ('blend', 'weighted_blend'):
    test_color_precision_shared_scene(alpha_mode, floating_precision='float32')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
