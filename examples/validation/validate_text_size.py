"""
Text size
=========

This example shows text with different sizes.

* On the left the text is in screen space. The size must match a
  reference, e.g. from text with the same font in a browser.
* On the right the text is in world space. The text must sit snugly with its
  baseline on the bottom line.
"""
# test_example = true
# sphinx_gallery_pygfx_render = True

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
scene = gfx.Scene()

obj0 = gfx.Text(
    gfx.TextGeometry(
        text="Screen  |  World", font_size=80, screen_space=True, anchor="bottom-center"
    ),
    gfx.TextMaterial(
        color="#aea",
        weight_offset=100,
        outline_color="#484",
        outline_thickness=0.2,
    ),
)
obj0.position.set(0, 50, 0)

obj1 = gfx.Text(
    gfx.TextGeometry(
        text="Lorem ipsum 40!", font_size=40, anchor="baseline-right", screen_space=True
    ),
    gfx.TextMaterial(color="#0ff"),
)
obj1.position.set(-10, 0, 0)

obj2 = gfx.Text(
    gfx.TextGeometry(
        text="Lorem ipsum 40!", font_size=40, anchor="baseline-left", screen_space=False
    ),
    gfx.TextMaterial(color="#0ff"),
)
obj2.position.set(10, 0, 0)

obj3 = gfx.Text(
    gfx.TextGeometry(
        text="Lorem ipsum 20 !",
        font_size=40,
        anchor="baseline-right",
        screen_space=True,
    ),
    gfx.TextMaterial(color="#0ff"),
)
obj3.position.set(-10, -50, 0)
obj3.scale.set(0.5, 0.5, 0.5)

obj4 = gfx.Text(
    gfx.TextGeometry(
        text="Lorem ipsum 20!", font_size=40, anchor="baseline-left", screen_space=False
    ),
    gfx.TextMaterial(color="#0ff"),
)
obj4.position.set(10, -50, 0)
obj4.scale.set(0.5, 0.5, 0.5)

obj5 = gfx.Text(
    gfx.TextGeometry(
        text="Rotated", font_size=20, anchor="baseline-right", screen_space=True
    ),
    gfx.TextMaterial(color="#0ff"),
)
obj5.position.set(-10, -100, 0)
obj5.rotation.set_from_axis_angle(gfx.linalg.Vector3(0, 0, 1), 0.2)

obj6 = gfx.Text(
    gfx.TextGeometry(
        text="Rotated", font_size=20, anchor="baseline-left", screen_space=False
    ),
    gfx.TextMaterial(color="#0ff"),
)
obj6.position.set(10, -100, 0)
obj6.rotation.set_from_axis_angle(gfx.linalg.Vector3(0, 0, 1), -0.2)

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
