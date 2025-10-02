"""
Test render-based picking.
"""

import pygfx as gfx
import numpy as np
import wgpu

render_tex = gfx.Texture(
    dim=2, size=(100, 100, 1), format=wgpu.TextureFormat.rgba8unorm
)


def test_render_picking_simple():
    """Render 3 different objects and pick them."""

    debug = False

    if debug:
        from rendercanvas.auto import RenderCanvas, loop

        renderer = gfx.WgpuRenderer(RenderCanvas())
    else:
        renderer = gfx.WgpuRenderer(render_tex)

    # Create a couple of objects

    ob1 = gfx.Mesh(
        gfx.plane_geometry(10, 10), gfx.MeshBasicMaterial(color="#f00", pick_write=True)
    )
    ob1.local.x = 5  # centered around 5 -> 0..10

    im = np.full((10, 10), 255, np.uint8)
    ob2 = gfx.Image(
        gfx.Geometry(grid=im),
        gfx.ImageBasicMaterial(pick_write=True),
    )
    ob2.local.x = 10.5  # origin at 10 -> 10..20
    ob2.local.y = -4.5

    ob3 = gfx.Line(
        gfx.Geometry(positions=[[25, -5, 0], [25, 5, 0]]),
        gfx.LineMaterial(
            thickness_space="model", thickness=10, color="#0f0", pick_write=True
        ),
    )

    # Put together, and create a camera that sees them
    scene = gfx.Scene().add(ob1, ob2, ob3)
    camera = gfx.OrthographicCamera()
    camera.show_rect(0, 30, -10, 10)

    if debug:
        renderer.request_draw(lambda: renderer.render(scene, camera))
        loop.run()
        return

    # Render and pick!
    renderer.render(scene, camera)

    info1 = renderer.get_pick_info((0 + 16, 50))
    info2 = renderer.get_pick_info((33 + 16, 50))
    info3 = renderer.get_pick_info((66 + 16, 50))

    # First assert colors
    assert info1["rgba"] == (1, 0, 0, 1)
    assert info2["rgba"] == (1, 1, 1, 1)
    assert info3["rgba"] == (0, 1, 0, 1)

    # Assert objects
    assert info1["world_object"] is ob1
    assert info2["world_object"] is ob2
    assert info3["world_object"] is ob3

    # Assert sub-info
    assert info1["face_index"] == 0
    assert np.allclose(info1["face_coord"], (0.0196, 0.487, 0.493), atol=0.001)

    assert info2["index"] == (4, 4)
    assert np.allclose(info2["pixel_coord"], (0.275, 0.425), atol=0.001)

    assert info3["vertex_index"] == 0
    assert abs(info3["segment_coord"] - 0.492) < 0.001


def test_render_picking_turning_on_off():
    """Turning picking on and off."""

    renderer = gfx.WgpuRenderer(render_tex)

    # Create a couple of objects

    ob1 = gfx.Mesh(gfx.plane_geometry(10, 10), gfx.MeshBasicMaterial(color="#f00"))
    ob1.local.x = 5  # centered around 5 -> 0..10

    # Put together, and create a camera that sees them
    scene = gfx.Scene().add(ob1)
    camera = gfx.OrthographicCamera()
    camera.show_rect(0, 10, -5, 5)

    # Render and pick!
    renderer.render(scene, camera)
    info1 = renderer.get_pick_info((50, 50))
    assert info1["rgba"] == (1, 0, 0, 1)
    assert info1["world_object"] is None  # no picking!

    # Turn picking on and try again
    ob1.material.pick_write = True

    renderer.render(scene, camera)
    info1 = renderer.get_pick_info((50, 50))
    assert info1["rgba"] == (1, 0, 0, 1)
    assert info1["world_object"] is ob1  # yay

    # Turn picking off again
    ob1.material.pick_write = False

    renderer.render(scene, camera)
    info1 = renderer.get_pick_info((50, 50))
    assert info1["rgba"] == (1, 0, 0, 1)
    assert info1["world_object"] is None


def test_render_picking_and_depth1():
    """Two pickable objects, the one in front wins."""

    # Note: to replicate this with a separate picking-pass,
    # that picking pass needs a depth buffer.

    renderer = gfx.WgpuRenderer(render_tex)

    # Create a couple of objects

    ob1 = gfx.Mesh(
        gfx.plane_geometry(10, 10), gfx.MeshBasicMaterial(color="#f00", pick_write=True)
    )
    ob1.local.x = 5  # centered around 5 -> 0..10

    ob2 = gfx.Mesh(
        gfx.plane_geometry(10, 10), gfx.MeshBasicMaterial(color="#0f0", pick_write=True)
    )
    ob2.local.x = 5  # centered around 5 -> 0..10

    # Put together, and create a camera that sees them
    scene = gfx.Scene().add(ob1, ob2)
    camera = gfx.OrthographicCamera()
    camera.show_rect(0, 10, -5, 5)

    # First, the red square is in front
    ob1.local.z = 1

    # Render and pick!
    renderer.render(scene, camera)
    info = renderer.get_pick_info((50, 50))
    assert info["rgba"] == (1, 0, 0, 1)  # red
    assert info["world_object"] is ob1

    # Now put the green square in front
    ob1.local.z = -1

    # Render and pick!
    renderer.render(scene, camera)
    info = renderer.get_pick_info((50, 50))
    assert info["rgba"] == (0, 1, 0, 1)  # green
    assert info["world_object"] is ob2


def test_render_picking_and_depth2():
    """One pickable objects can be obscured by a non-pickable object."""

    # Note: if we'd do picking in a separate render pass,
    # the obscuring would not happen.
    # One can argue about what is the preferred outcome.

    renderer = gfx.WgpuRenderer(render_tex)

    # Create a couple of objects

    ob1 = gfx.Mesh(
        gfx.plane_geometry(10, 10),
        gfx.MeshBasicMaterial(
            color="#f00", alpha_mode="blend", depth_write=True, pick_write=True
        ),
    )
    ob1.local.x = 5  # centered around 5 -> 0..10

    ob2 = gfx.Mesh(
        gfx.plane_geometry(10, 10),
        gfx.MeshBasicMaterial(
            color="#0f0", alpha_mode="blend", depth_write=True, pick_write=False
        ),
    )
    ob2.local.x = 5  # centered around 5 -> 0..10

    # Put together, and create a camera that sees them
    scene = gfx.Scene().add(ob1, ob2)
    camera = gfx.OrthographicCamera()
    camera.show_rect(0, 10, -5, 5)

    # First, the red square is in front
    ob1.local.z = 1

    # Render and pick!
    renderer.render(scene, camera)
    info = renderer.get_pick_info((50, 50))
    assert info["rgba"] == (1, 0, 0, 1)  # red
    assert info["world_object"] is ob1  # makes sense

    # Now put the green square in front
    ob1.local.z = -1

    # Render and pick!
    renderer.render(scene, camera)
    info = renderer.get_pick_info((50, 50))
    assert info["rgba"] == (0, 1, 0, 1)  # green
    assert info["world_object"] is ob1  # huh?

    # The thing is that the red square is rendered first, writing
    # to both the color and pick buffer. Then the green square is drawn,
    # but it only draws to the color buffer.

    # If we mark the objects as opaque, the renderer will sort them front-to-back
    # to avoid overdraw, causing green to be rendered first, and then
    # the red fragments dont event hit the fragment shader.
    ob1.material.render_queue = 2000
    ob2.material.render_queue = 2000

    # Render and pick!
    renderer.render(scene, camera)
    info = renderer.get_pick_info((50, 50))
    assert info["rgba"] == (0, 1, 0, 1)  # green
    assert info["world_object"] is None  # works!


def test_render_picking_and_transparency():
    """Transparent objects are pickable."""

    # Note: if we'd do picking in a separate render pass,
    # the obscuring would not happen.
    # One can argue about what is the preferred outcome.

    renderer = gfx.WgpuRenderer(render_tex)

    # Create a couple of objects

    ob1 = gfx.Mesh(
        gfx.plane_geometry(10, 10),
        gfx.MeshBasicMaterial(color="#f00", opacity=0.2, pick_write=True),
    )
    ob1.local.x = 5  # centered around 5 -> 0..10

    ob2 = gfx.Mesh(
        gfx.plane_geometry(10, 10),
        gfx.MeshBasicMaterial(color="#0f0", pick_write=False),
    )
    ob2.local.x = 5  # centered around 5 -> 0..10

    # Put together, and create a camera that sees them
    scene = gfx.Scene().add(ob1, ob2)
    camera = gfx.OrthographicCamera()
    camera.show_rect(0, 10, -5, 5)

    # The red square is in front
    ob1.local.z = 1

    # Render and pick!
    # We don't check the color, because now its mixed red/green
    renderer.render(scene, camera)
    info = renderer.get_pick_info((50, 50))
    assert info["world_object"] is ob1

    # It does not matter whether the green object is pickable or not
    ob2.material.pick_write = True

    renderer.render(scene, camera)
    info = renderer.get_pick_info((50, 50))
    assert info["world_object"] is ob1

    # And it also works with fully transparent objects
    ob1.material.opacity = 0.0

    renderer.render(scene, camera)
    info = renderer.get_pick_info((50, 50))
    assert info["world_object"] is ob1

    # Except off-course if you also use an alpa test :)
    ob1.material.alpha_test = 0.01

    renderer.render(scene, camera)
    info = renderer.get_pick_info((50, 50))
    assert info["world_object"] is ob2


if __name__ == "__main__":
    test_render_picking_simple()
    test_render_picking_turning_on_off()
    test_render_picking_and_depth1()
    test_render_picking_and_depth2()
    test_render_picking_and_transparency()
