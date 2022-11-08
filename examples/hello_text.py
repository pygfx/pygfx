"""
Example showing text.
"""

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
scene = gfx.Scene()


gfx.utils.text.font_manager.add_font_file(
    "/Users/almar/dev/py/pygfx/pygfx/pkg_resources/fonts/NotoSans-Bold.ttf"
)

obj = gfx.Text(
    # gfx.text_geometry(text="", font_size=70),
    gfx.text_geometry(
        text="Hello world🥳 مرحبا بالعالم", font_size=60, family="NotoSans-Bold"
    ),
    gfx.TextMaterial(
        color="#444",
        outline_color="#000",
        screen_space=True,
        aa=True,
        outline_thickness=0.0,
    ),
)
scene.add(obj)

obj2 = gfx.Text(
    # gfx.text_geometry(text="", font_size=70),
    gfx.text_geometry(
        text="Hello world🥳 مرحبا بالعالم", font_size=60, weight="regular"
    ),
    gfx.TextMaterial(
        color="#444",
        outline_color="#000",
        screen_space=True,
        aa=True,
        outline_thickness=0.0,
    ),
)
scene.add(obj2)
obj2.position.z = 40


scene.add(gfx.Background(None, gfx.BackgroundMaterial("#fff")))
scene.add(gfx.AxesHelper(size=250))

camera = gfx.PerspectiveCamera(70, 1)
camera.position.z = 30

controller = gfx.OrbitController(camera.position.clone())
controller.add_default_event_handlers(renderer, camera)


def animate():
    controller.update_camera(camera)
    renderer.render(scene, camera)


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
