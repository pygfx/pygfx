import pytest
import wgpu

import pygfx as gfx

from ..testutils import can_use_wgpu_lib

if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


def _stencil_scene(reader_value):
    """A writer marks stencil=1 on the left half; a full-screen white reader
    draws only where the stencil buffer equals its reference value."""
    scene = gfx.Scene()
    state = wgpu.StencilFaceState(compare="always", pass_op="replace")
    writer = gfx.Mesh(
        gfx.plane_geometry(1, 2),
        gfx.MeshBasicMaterial(
            color="#000",
            depth_test=False,
            stencil_value=1,
            stencil_front_state=state,
            stencil_back_state=state,
        ),
    )
    writer.local.x = -0.5
    writer.render_order = 0
    state = wgpu.StencilFaceState(compare="equal")
    reader = gfx.Mesh(
        gfx.plane_geometry(2, 2),
        gfx.MeshBasicMaterial(
            color="#fff",
            depth_test=False,
            stencil_value=reader_value,
            stencil_front_state=state,
            stencil_back_state=state,
        ),
    )
    reader.render_order = 1
    scene.add(writer, reader)
    return scene, reader


def _render(renderer, scene):
    renderer.render(scene, gfx.NDCCamera())
    im = renderer.snapshot()
    h, w = im.shape[:2]
    return int(im[h // 2, w // 4, 0]), int(im[h // 2, 3 * w // 4, 0])


def test_stencil_value_change_is_applied():
    texture = gfx.Texture(dim=2, size=(64, 48, 1), format="rgba8unorm-srgb")
    renderer = gfx.renderers.WgpuRenderer(texture, enable_stencil=True)
    renderer.set_clear_value(stencil=0)
    scene, reader = _stencil_scene(0)

    left, right = _render(renderer, scene)
    assert left == 0 and right > 200  # reader passes where stencil == 0

    # Changing only the reference value must take effect on the next draw
    reader.material.stencil_value = 1
    left, right = _render(renderer, scene)
    assert left > 200 and right == 0
    assert reader.material.stencil_value == 1
