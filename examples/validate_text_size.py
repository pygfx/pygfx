"""
This example shows text with different sizes.

* On the left the text is in screen space. The size must match a
  reference, e.g. from text with the same font in a browser.
* On the right the text is in world space. The text must fit nicely in
  between the green lines.
"""
# test_example = true

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


# todo: text anchor center would be nice here

renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
scene = gfx.Scene()

obj0 = gfx.Text(
    gfx.text_geometry(text="Screen  |   World", font_size=80),
    gfx.TextMaterial(
        color="#555",
        screen_space=True,
    ),
)
obj0.position.set(-230, 50, 0)

obj1 = gfx.Text(
    gfx.text_geometry(text="Lorem ipsum!", font_size=40),
    gfx.TextMaterial(
        color="#fff",
        screen_space=True,
    ),
)
obj1.position.set(-250, 0, 0)

obj2 = gfx.Text(
    gfx.text_geometry(text="Lorem ipsum!", font_size=40),
    gfx.TextMaterial(
        color="#fff",
        screen_space=False,
    ),
)
obj2.position.set(0, 0, 0)

obj3 = gfx.Text(
    gfx.text_geometry(text="Lorem ipsum! (screen small)", font_size=20),
    gfx.TextMaterial(
        color="#fff",
        screen_space=True,
    ),
)
obj3.position.set(-250, -50, 0)
obj3.scale.set(
    0.5, 0.5, 0.5
)  # This (intentionally) does not work, we set font_size instead

obj4 = gfx.Text(
    gfx.text_geometry(text="Lorem ipsum! (world small)", font_size=40),
    gfx.TextMaterial(
        color="#fff",
        screen_space=False,
    ),
)
obj4.position.set(0, -50, 0)
obj4.scale.set(0.5, 0.5, 0.5)

line = gfx.Line(
    gfx.Geometry(positions=[(0, 0, 0), (900, 0, 0), (0, 40, 0), (900, 40, 0)]),
    gfx.LineSegmentMaterial(color="green"),
)

scene.add(line, obj0, obj1, obj2, obj3, obj4)


camera = gfx.OrthographicCamera(500, 400)
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
