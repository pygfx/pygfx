"""
Example showing text.
"""

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
scene = gfx.Scene()

obj = gfx.Text(
    gfx.text_geometry(text="مرحبا بالعالم", font_size=70),
    # gfx.text_geometry(text="Hello world fi", font_size=70),
    gfx.TextMaterial(
        color="#444",
        outline_color="#000",
        screen_space=True,
        aa=True,
        outline_thickness=0.0,
    ),
)
scene.add(obj)

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
