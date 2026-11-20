"""Tests for the depth bias properties of Material."""

import pygfx as gfx
import pytest

from ..testutils import can_use_wgpu_lib

if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


render_tex = gfx.Texture(dim=2, size=(64, 64, 1), format="rgba8unorm")


def test_depth_bias():
    """With two coplanar planes, a bias on one decides which is visible."""

    red = gfx.Mesh(gfx.plane_geometry(2, 2), gfx.MeshBasicMaterial(color="#f00"))
    green = gfx.Mesh(gfx.plane_geometry(2, 2), gfx.MeshBasicMaterial(color="#0f0"))
    green.render_order = 1  # drawn after red, so it loses the "<" depth test

    scene = gfx.Scene().add(red, green)
    camera = gfx.OrthographicCamera(2, 2)
    renderer = gfx.WgpuRenderer(render_tex)

    def center_color():
        renderer.render(scene, camera)
        return renderer.get_pick_info((32, 32))["rgba"]

    assert center_color() == (1, 0, 0, 1)

    # Pushing red away from the camera makes green win the depth test
    red.material.depth_bias = 16
    assert center_color() == (0, 1, 0, 1)

    red.material.depth_bias = 0
    assert center_color() == (1, 0, 0, 1)


def test_depth_bias_slope_scale():
    """The slope-scale bias is applied (tilted planes, constant bias zero)."""

    red = gfx.Mesh(gfx.plane_geometry(2, 2), gfx.MeshBasicMaterial(color="#f00"))
    green = gfx.Mesh(gfx.plane_geometry(2, 2), gfx.MeshBasicMaterial(color="#0f0"))
    green.render_order = 1
    red.local.euler_x = green.local.euler_x = 0.8

    scene = gfx.Scene().add(red, green)
    camera = gfx.OrthographicCamera(2, 2)
    renderer = gfx.WgpuRenderer(render_tex)

    def center_color():
        renderer.render(scene, camera)
        return renderer.get_pick_info((32, 32))["rgba"]

    assert center_color() == (1, 0, 0, 1)

    red.material.depth_bias_slope_scale = 2.0
    assert center_color() == (0, 1, 0, 1)


if __name__ == "__main__":
    test_depth_bias()
    test_depth_bias_slope_scale()
