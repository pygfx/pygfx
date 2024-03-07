import logging

import numpy as np
import wgpu
import pygfx as gfx

# from ..testutils import can_use_wgpu_lib
import pytest


# if not can_use_wgpu_lib:
#     pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


render_tex = gfx.Texture(dim=2, size=(10, 10, 1), format=wgpu.TextureFormat.rgba8unorm)
renderer1 = gfx.renderers.WgpuRenderer(render_tex)
renderer2 = gfx.renderers.WgpuRenderer(render_tex)
renderer1.blend_mode = "ordered1"
renderer2.blend_mode = "weighted"

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.local.position = (0, 0, 400)


class Handler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def render(wobject, renderer=renderer1):
    # Setup logging
    h = Handler()
    gfx.logger.addHandler(h)
    gfx.logger.setLevel("INFO")
    # Render
    renderer.render(wobject, camera)
    # Detach logging
    gfx.logger.removeHandler(h)
    # Detect changed labels
    changed = set()
    for record in h.records:
        text = record.message
        _, _, sub = text.partition("shader update:")
        changed.update(sub.strip(".").replace(",", " ").split())
    return changed


def test_reactivity_mesh1():
    # Test basics

    # Prepare

    cube = gfx.Mesh(
        gfx.box_geometry(200, 200, 200),
        gfx.MeshPhongMaterial(color="#336699"),
    )

    # Render once
    render(cube)

    # Changing the color should not change anything
    cube.material.color = "red"
    changed = render(cube)
    assert changed == set()

    # Changing the render mask requires new render info
    cube.render_mask = "all"
    changed = render(cube)
    assert changed == {"render_info"}

    # Changing the side requires a new pipeline
    cube.material.side = "FRONT"
    changed = render(cube)
    assert changed == {"pipeline_info", "render_info", "compose_pipeline"}

    # Changing the wireframe requires a new shader
    cube.material.wireframe = True
    changed = render(cube)
    assert "create" in changed


def test_reactivity_mesh2():
    # Test swapping bindings

    # Prepare

    cube = gfx.Mesh(
        gfx.box_geometry(200, 200, 200),
        gfx.MeshPhongMaterial(color="#336699"),
    )

    m1 = cube.material  # noqa
    m2 = gfx.MeshPhongMaterial(color="#ff6699")
    m3 = gfx.MeshBasicMaterial(color="#11ff99")

    p1 = cube.geometry.positions
    p2 = gfx.Buffer(p1.data * 0.7)

    g1 = cube.geometry
    g2 = gfx.Geometry()
    g2.positions = p2
    g2.normals = g1.normals
    g2.indices = g1.indices

    # Render once
    render(cube)

    # Swap out the material - same type
    cube.material = m2
    changed = render(cube)
    assert changed == {"bindings"}

    # Swap out the material - different type
    cube.material = m3
    changed = render(cube)
    assert "create" in changed
    assert "compile_shader" in changed
    assert "compose_pipeline" in changed

    # Swap out the positions
    cube.geometry.positions = p2
    changed = render(cube)
    assert changed == {"bindings"}

    # Swap out the whole geometry
    cube.geometry = g2
    changed = render(cube)
    assert changed == {"bindings"}


def test_reactivity_mesh3():
    # Test swapping colormap

    geometry = gfx.torus_knot_geometry(1, 0.3, 128, 32)
    geometry.texcoords = gfx.Buffer(geometry.texcoords.data[:, 0].copy())

    tex1 = gfx.cm.cividis
    tex2 = gfx.cm.inferno
    cmap3 = np.array([(1,), (0,), (0,), (1,)], np.int32)
    tex3 = gfx.Texture(cmap3, dim=1)

    # only float32 color map is supported in MeshPhongMaterial for now
    obj = gfx.Mesh(geometry, gfx.MeshBasicMaterial(map=tex1))

    # Render once
    render(obj)

    # Change to a colormap with the same format, all is ok!
    obj.material.map = tex2
    changed = render(obj)
    assert changed == {"bindings"}

    # Change to colormap of different format, need rebuild!
    obj.material.map = tex3
    print("uv", geometry.texcoords.data.shape)
    print("map", obj.material.map.dim)
    changed = render(obj)
    assert changed == {"bindings", "compile_shader", "compose_pipeline"}


def test_change_blend_mode():
    cube = gfx.Mesh(
        gfx.box_geometry(200, 200, 200),
        gfx.MeshPhongMaterial(color="#336699"),
    )

    # Render once
    renderer1.blend_mode = "ordered1"
    changed = render(cube, renderer1)
    assert "create" in changed

    # Render with another blend mode
    renderer1.blend_mode = "ordered2"
    changed = render(cube, renderer1)
    assert changed == {"compose_pipeline", "compile_shader"}

    # Render in the first again
    # The fact that it recompiles is an indication that the
    # environment-specific wgpu objects were cleaned up.
    renderer1.blend_mode = "ordered1"
    changed = render(cube, renderer1)
    assert changed == {"compose_pipeline", "compile_shader"}

    # Setting the blend_mode to the same current value should not trigger
    renderer1.blend_mode = "ordered1"
    changed = render(cube, renderer1)
    assert changed == set()


def test_two_renders_with_same_blend_modes():
    cube = gfx.Mesh(
        gfx.box_geometry(200, 200, 200),
        gfx.MeshPhongMaterial(color="#336699"),
    )

    # Render once
    renderer1.blend_mode = "ordered1"
    changed = render(cube, renderer1)
    assert "create" in changed

    # Render in another renderer, with same blend mode!
    renderer2.blend_mode = "ordered1"
    changed = render(cube, renderer2)
    assert changed == set()


def test_two_renders_with_different_blend_modes():
    cube = gfx.Mesh(
        gfx.box_geometry(200, 200, 200),
        gfx.MeshPhongMaterial(color="#336699"),
    )

    # Render once
    renderer1.blend_mode = "ordered1"
    changed = render(cube, renderer1)
    assert "create" in changed

    # Render in another renderer
    renderer2.blend_mode = "ordered2"
    changed = render(cube, renderer2)
    assert changed == {"compose_pipeline", "compile_shader"}


if __name__ == "__main__":
    pytest.main(["-x", __file__])
