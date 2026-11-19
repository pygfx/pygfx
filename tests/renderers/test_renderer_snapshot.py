"""Tests for WgpuRenderer.snapshot(), covering every branch of the
source-selection if-statement and every branch of the texture-format
if-statement in its implementation.
"""

import numpy as np
import pytest
import wgpu

import pygfx as gfx

from ..testutils import can_use_wgpu_lib


if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


SIZE = (16, 16)


def make_renderer(**kwargs):
    target = gfx.Texture(dim=2, size=(*SIZE, 1), format="rgba8unorm")
    target._wgpu_usage |= wgpu.TextureUsage.COPY_SRC
    renderer = gfx.WgpuRenderer(target, pixel_scale=1, **kwargs)
    return renderer


def render_plane(renderer, **material_kwargs):
    material = gfx.MeshBasicMaterial(color="#f00", pick_write=True, **material_kwargs)
    mesh = gfx.Mesh(gfx.plane_geometry(4, 4), material)
    scene = gfx.Scene().add(mesh)
    camera = gfx.OrthographicCamera(2, 2)
    camera.show_rect(-4, 4, -4, 4)  # plane covers the center, corners stay background
    renderer.render(scene, camera)
    return renderer


# -- source selection (first if-statement) --


def test_snapshot_color_default():
    # _name_of_texture_with_effects is None before any flush; falls back to "color"
    renderer = make_renderer()
    render_plane(renderer)
    renderer._name_of_texture_with_effects = None
    image = renderer.snapshot("color")
    assert image.shape == (*SIZE[::-1], 4)
    assert image.dtype == np.uint8
    # plane is red, center pixel should be reddish
    assert image[8, 8, 0] > 200
    assert image[8, 8, 1] < 60


def test_snapshot_color_after_effects():
    # with an effect pass, the color snapshot reads the ping-ponged "altcolor" texture
    renderer = make_renderer(ppaa="fxaa")
    render_plane(renderer)
    assert renderer._name_of_texture_with_effects == "altcolor"
    image = renderer.snapshot("color")
    assert image.shape == (*SIZE[::-1], 4)
    assert image.dtype == np.uint8


def test_snapshot_depth_and_stencil_select_depth_stencil_texture():
    # both "depth" and "stencil" map to the "depth_stencil" blender texture;
    # use enable_stencil=True so that target is depth32float-stencil8
    renderer = make_renderer(enable_stencil=True)
    render_plane(renderer)
    depth = renderer.snapshot("depth")
    stencil = renderer.snapshot("stencil")
    assert depth.shape == SIZE[::-1]
    assert depth.dtype == np.float32
    assert stencil.shape == SIZE[::-1]
    assert stencil.dtype == np.uint8
    # depth is cleared to 1.0, the plane is closer
    assert depth[8, 8] < 1.0
    assert depth[0, 0] == 1.0
    # stencil is cleared to 0
    assert (stencil == 0).all()


def test_snapshot_pick():
    renderer = make_renderer()
    render_plane(renderer)  # material has pick_write=True, so the pick texture exists
    pick = renderer.snapshot("pick")
    assert pick.shape == (*SIZE[::-1], 4)
    assert pick.dtype == np.uint16


def test_snapshot_invalid_source():
    renderer = make_renderer()
    render_plane(renderer)
    with pytest.raises(ValueError, match="Unknown snapshot source"):
        renderer.snapshot("nonsense")


# -- texture format branches (second if-statement) --


def test_snapshot_format_rgba8unorm():
    # default color target is rgba8unorm-srgb; swap in a plain rgba8unorm texture
    renderer = make_renderer()
    render_plane(renderer)
    renderer._name_of_texture_with_effects = None  # make snapshot read "color"
    blender = renderer._blender
    tex = blender.device.create_texture(
        size=(*SIZE, 1),
        format="rgba8unorm",
        usage=wgpu.TextureUsage.COPY_SRC,
    )
    blender._textures["color"] = tex
    image = renderer.snapshot("color")
    assert image.shape == (*SIZE[::-1], 4)
    assert image.dtype == np.uint8


def test_snapshot_format_rgba16float():
    renderer = make_renderer(color_precision="float16")
    render_plane(renderer)
    image = renderer.snapshot("color")
    assert image.shape == (*SIZE[::-1], 4)
    assert image.dtype == np.float16


def test_snapshot_format_rgba32float():
    try:
        renderer = make_renderer(color_precision="float32")
    except RuntimeError as e:
        pytest.skip(f"float32 color precision not supported: {e}")
    render_plane(renderer)
    image = renderer.snapshot("color")
    assert image.shape == (*SIZE[::-1], 4)
    assert image.dtype == np.float32


def test_snapshot_format_rgba32float_via_texture_swap():
    # rgba32float color target (color_precision="float32") needs float32-blendable,
    # which not all adapters have; swap in an rgba32float texture directly instead
    renderer = make_renderer()
    render_plane(renderer)
    renderer._name_of_texture_with_effects = None  # make snapshot read "color"
    blender = renderer._blender
    tex = blender.device.create_texture(
        size=(*SIZE, 1),
        format="rgba32float",
        usage=wgpu.TextureUsage.COPY_SRC,
    )
    blender._textures["color"] = tex
    image = renderer.snapshot("color")
    assert image.shape == (*SIZE[::-1], 4)
    assert image.dtype == np.float32


def test_snapshot_format_rgba16uint():
    # the pick texture is rgba16uint; covered via source="pick"
    renderer = make_renderer()
    render_plane(renderer)
    pick = renderer.snapshot("pick")
    assert pick.dtype == np.uint16
    assert pick.shape == (*SIZE[::-1], 4)


def test_snapshot_format_stencil8():
    # stencil-only target
    renderer = make_renderer(enable_depth=False, enable_stencil=True)
    render_plane(
        renderer,
        stencil_value=1,
        stencil_front_state=wgpu.StencilFaceState(compare="always", pass_op="replace"),
    )
    stencil = renderer.snapshot("stencil")
    assert stencil.shape == SIZE[::-1]
    assert stencil.dtype == np.uint8
    # the plane wrote stencil=1 where it is drawn
    assert stencil[8, 8] == 1
    assert stencil[0, 0] == 0


def test_snapshot_format_depth32float():
    # depth-only target
    renderer = make_renderer(enable_stencil=False)
    render_plane(renderer)
    depth = renderer.snapshot("depth")
    assert depth.shape == SIZE[::-1]
    assert depth.dtype == np.float32
    assert depth[8, 8] < 1.0
    assert depth[0, 0] == 1.0


def test_snapshot_format_depth32float_stencil8():
    # combined target: both the depth and the stencil sub-branch
    renderer = make_renderer(enable_stencil=True)
    render_plane(
        renderer,
        stencil_value=1,
        stencil_front_state=wgpu.StencilFaceState(compare="always", pass_op="replace"),
    )
    depth = renderer.snapshot("depth")
    stencil = renderer.snapshot("stencil")
    assert depth.dtype == np.float32
    assert depth.shape == SIZE[::-1]
    assert stencil.dtype == np.uint8
    assert stencil.shape == SIZE[::-1]
    assert stencil[8, 8] == 1
    assert stencil[0, 0] == 0


def test_snapshot_format_unexpected_raises():
    renderer = make_renderer()
    render_plane(renderer)
    renderer._name_of_texture_with_effects = None  # make snapshot read "color"
    blender = renderer._blender
    tex = blender.device.create_texture(
        size=(*SIZE, 1),
        format="bgra8unorm",
        usage=wgpu.TextureUsage.COPY_SRC,
    )
    blender._textures["color"] = tex
    with pytest.raises(RuntimeError, match="Unexpected format for snapshot"):
        renderer.snapshot("color")
