"""
This example shows text with different sizes.

* On the left the text is in screen space. The size must match a
  reference, e.g. from text with the same font in a browser.
* On the right the text is in world space. The text must sit snugly with its
  baseline on the bottom line.
"""
# test_example = true

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
scene = gfx.Scene()

obj0 = gfx.Text(
    gfx.TextGeometry(text="Screen  |  World", font_size=80, anchor="bottom-center"),
    gfx.TextMaterial(
        color="#aea",
        screen_space=True,
        weight_offset=100,
        outline_color="#484",
        outline_thickness=0.2,
    ),
)
obj0.position.set(0, 50, 0)

obj1 = gfx.Text(
    gfx.TextGeometry(text="Lorem ipsum 40!", font_size=40, anchor="baseline-right"),
    gfx.TextMaterial(color="#fff", screen_space=True),
)
obj1.position.set(-10, 0, 0)

obj2 = gfx.Text(
    gfx.TextGeometry(text="Lorem ipsum 40!", font_size=40, anchor="baseline-left"),
    gfx.TextMaterial(color="#fff", screen_space=False),
)
obj2.position.set(10, 0, 0)

obj3 = gfx.Text(
    gfx.TextGeometry(text="Lorem ipsum 20 !", font_size=40, anchor="baseline-right"),
    gfx.TextMaterial(color="#fff", screen_space=True),
)
obj3.position.set(-10, -50, 0)
obj3.scale.set(0.5, 0.5, 0.5)

obj4 = gfx.Text(
    gfx.TextGeometry(text="Lorem ipsum 20!", font_size=40, anchor="baseline-left"),
    gfx.TextMaterial(color="#fff", screen_space=False),
)
obj4.position.set(10, -50, 0)
obj4.scale.set(0.5, 0.5, 0.5)

obj5 = gfx.Text(
    gfx.TextGeometry(text="Rotated", font_size=20, anchor="baseline-right"),
    gfx.TextMaterial(color="#fff", screen_space=True),
)
obj5.position.set(-10, -100, 0)
obj5.rotation.set_from_axis_angle(gfx.linalg.Vector3(0, 0, 1), 0.2)

obj6 = gfx.Text(
    gfx.TextGeometry(text="Rotated", font_size=20, anchor="baseline-left"),
    gfx.TextMaterial(color="#fff", screen_space=False),
)
obj6.position.set(10, -100, 0)
obj6.rotation.set_from_axis_angle(gfx.linalg.Vector3(0, 0, 1), -0.2)

line = gfx.Line(
    gfx.Geometry(positions=[(0, 0, 0), (900, 0, 0), (0, 40, 0), (900, 40, 0)]),
    gfx.LineSegmentMaterial(color="green"),
)

scene.add(line, obj0, obj1, obj2, obj3, obj4, obj5, obj6)


camera = gfx.OrthographicCamera(650, 400)
camera.position.z = 30

controller = gfx.OrbitController(camera.position.clone())
controller.add_default_event_handlers(renderer, camera)


def animate():
    controller.update_camera(camera)
    renderer.render(scene, camera)


renderer.request_draw(animate)

if __name__ == "__main__":
    print(__doc__)
    run()