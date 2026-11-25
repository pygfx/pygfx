"""Tests for Material.color_write."""

import pytest

import pygfx as gfx

from ..testutils import can_use_wgpu_lib


if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


target = gfx.Texture(dim=2, size=(64, 64, 1), format="rgba8unorm")


def test_color_write_depth_only():
    front = gfx.Mesh(
        gfx.plane_geometry(2, 2), gfx.MeshBasicMaterial(color="#f00", color_write=False)
    )
    behind = gfx.Mesh(gfx.plane_geometry(1, 1), gfx.MeshBasicMaterial(color="#0f0"))
    behind.local.z = -1
    behind.render_order = 1  # drawn after the front plane
    scene = gfx.Scene().add(front, behind)
    camera = gfx.OrthographicCamera(2, 2)
    renderer = gfx.WgpuRenderer(target)

    def center_color():
        renderer.render(scene, camera)
        return renderer.get_pick_info((32, 32))["rgba"]

    assert center_color() == (0, 0, 0, 0)  # invisible, but hides the green plane
    front.material.color_write = True
    assert center_color() == "#f00"
    front.material.color_write = False
    front.material.depth_write = False
    assert center_color() == "#0f0"


@pytest.mark.parametrize("alpha_mode", ["solid", "dither", "blend", "weighted_blend"])
def test_color_write_invisible_but_pickable(alpha_mode):
    ob = gfx.Mesh(
        gfx.plane_geometry(2, 2),
        gfx.MeshBasicMaterial(
            color="#f00", alpha_mode=alpha_mode, color_write=False, pick_write=True
        ),
    )
    renderer = gfx.WgpuRenderer(target)
    renderer.render(gfx.Scene().add(ob), gfx.OrthographicCamera(2, 2))
    info = renderer.get_pick_info((32, 32))
    assert info["rgba"] == (0, 0, 0, 0)
    assert info["world_object"] is ob
