"""Tests for Material.color_write."""

import numpy as np
import pytest

import pygfx as gfx

from ..testutils import can_use_wgpu_lib


def test_color_write_prop():
    m = gfx.MeshBasicMaterial()
    assert m.color_write is True
    m.color_write = False
    assert m.color_write is False
    m2 = gfx.MeshBasicMaterial(color_write=False)
    assert m2.color_write is False
    with pytest.raises(TypeError):
        m.color_write = "nope"


@pytest.mark.skipif(not can_use_wgpu_lib, reason="Needs the wgpu lib")
def test_color_write_depth_only_draw():
    """An object with color_write=False is invisible but still writes depth,
    hiding objects behind it."""
    target = gfx.Texture(dim=2, size=(64, 64, 1), format="rgba8unorm")
    renderer = gfx.renderers.WgpuRenderer(target, ppaa="none", pixel_ratio=1)

    scene = gfx.Scene()
    scene.add(gfx.Background.from_color("#000"))
    hidden = gfx.Mesh(
        gfx.plane_geometry(2, 2),
        gfx.MeshBasicMaterial(color="#ff0000", color_write=False),
    )
    hidden.local.position = (0, 0, 1)
    behind = gfx.Mesh(
        gfx.plane_geometry(0.8, 0.8),
        gfx.MeshBasicMaterial(color="#00ff00"),
    )
    behind.render_order = 1  # drawn after, but behind the depth-only plane
    scene.add(hidden, behind)

    camera = gfx.PerspectiveCamera(50)
    camera.local.position = (0, 0, 3)

    renderer.render(scene, camera)
    img = np.asarray(renderer.snapshot())
    red = ((img[..., 0] > 128) & (img[..., 1] < 64)).sum()
    green = ((img[..., 1] > 128) & (img[..., 0] < 64)).sum()
    assert red == 0  # the color_write=False plane is invisible
    assert green == 0  # but its depth still occludes the plane behind it

    # With color_write on, the front plane shows up
    hidden.material.color_write = True
    renderer.render(scene, camera)
    img = np.asarray(renderer.snapshot())
    red = ((img[..., 0] > 128) & (img[..., 1] < 64)).sum()
    assert red > 100


@pytest.mark.skipif(not can_use_wgpu_lib, reason="Needs the wgpu lib")
@pytest.mark.parametrize("alpha_mode", ["solid", "dither", "blend", "weighted_blend"])
def test_color_write_invisible_but_pickable(alpha_mode):
    """With color_write=False nothing is drawn, for each alpha method, while
    pick_write still works."""
    target = gfx.Texture(dim=2, size=(64, 64, 1), format="rgba8unorm")
    renderer = gfx.renderers.WgpuRenderer(target, ppaa="none", pixel_ratio=1)

    ob = gfx.Mesh(
        gfx.plane_geometry(2, 2),
        gfx.MeshBasicMaterial(
            color="#f00", alpha_mode=alpha_mode, color_write=False, pick_write=True
        ),
    )
    camera = gfx.OrthographicCamera()
    camera.show_rect(-1, 1, -1, 1)

    renderer.render(gfx.Scene().add(ob), camera)
    info = renderer.get_pick_info((32, 32))
    assert info["rgba"] == (0, 0, 0, 0)
    assert info["world_object"] is ob


if __name__ == "__main__":
    pytest.main(["-x", __file__])
