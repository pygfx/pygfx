"""
Text size
=========

This example shows text with different sizes.

* On the left the text is in screen space. The size must match a
  reference, e.g. from text with the same font in a browser.
* On the right the text is in world space. The text must sit snugly with its
  baseline on the bottom line.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
scene = gfx.Scene()

obj0 = gfx.Text(
    text="Screen  |  World",
    font_size=80,
    screen_space=True,
    anchor="bottom-center",
    material=gfx.TextMaterial(
        color="#aea",
        weight_offset=100,
        outline_color="#484",
        outline_thickness=0.2,
    ),
)
obj0.local.position = (0, 50, 0)

obj1 = gfx.Text(
    text="Lorem ipsum 40!",
    font_size=40,
    anchor="baseline-right",
    screen_space=True,
    material=gfx.TextMaterial(color="#0ff"),
)
obj1.local.position = (-10, 0, 0)

obj2 = gfx.Text(
    text="Lorem ipsum 40!",
    font_size=40,
    anchor="baseline-left",
    screen_space=False,
    material=gfx.TextMaterial(color="#0ff"),
)
obj2.local.position = (10, 0, 0)

obj3 = gfx.Text(
    text="Lorem ipsum 20 !",
    font_size=40,
    anchor="baseline-right",
    screen_space=True,
    material=gfx.TextMaterial(color="#0ff"),
)
obj3.local.position = (-10, -50, 0)
obj3.local.scale = (0.5, 0.5, 0.5)

obj4 = gfx.Text(
    text="Lorem ipsum 20!",
    font_size=40,
    anchor="baseline-left",
    screen_space=False,
    material=gfx.TextMaterial(color="#0ff"),
)
obj4.local.position = (10, -50, 0)
obj4.local.scale = (0.5, 0.5, 0.5)

obj5 = gfx.Text(
    text="Rotated",
    font_size=20,
    anchor="baseline-right",
    screen_space=True,
    material=gfx.TextMaterial(color="#0ff"),
)
obj5.local.position = (-10, -100, 0)
obj5.local.rotation = la.quat_from_axis_angle((0, 0, 1), 0.2)

obj6 = gfx.Text(
    text="Rotated",
    font_size=20,
    anchor="baseline-left",
    screen_space=False,
    material=gfx.TextMaterial(color="#0ff"),
)
obj6.local.position = (10, -100, 0)
obj6.local.rotation = la.quat_from_axis_angle((0, 0, 1), -0.2)

line = gfx.Line(
    gfx.Geometry(positions=[(0, 0, 0), (900, 0, 0), (0, 40, 0), (900, 40, 0)]),
    gfx.LineSegmentMaterial(color="green"),
)

scene.add(line, obj0, obj1, obj2, obj3, obj4, obj5, obj6)

camera = gfx.OrthographicCamera()
camera.show_rect(-325, 325, -200, 200)
controller = gfx.OrbitController(camera, register_events=renderer)


renderer.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    run()
