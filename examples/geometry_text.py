"""
Example showing text.
"""

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
scene = gfx.Scene()

obj = gfx.Text(
    gfx.text_geometry(text="Hello world"),
    gfx.TextMaterial(color="cyan"),
)
scene.add(obj)

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
