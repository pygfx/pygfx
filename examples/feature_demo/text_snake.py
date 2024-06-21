"""
Text snake
==========
Example showing a text with picking and custom layout.
"""

# sphinx_gallery_pygfx_docs = 'animate 3s'
# sphinx_gallery_pygfx_test = 'run'


from time import perf_counter

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
scene = gfx.Scene()


class SnakeTextGeometry(gfx.TextGeometry):
    def _apply_layout(self):
        super()._apply_layout()

        for i in range(self.positions.nitems):
            x, y = self.positions.data[i]
            y += 8 * np.sin(x * 0.1 + perf_counter() * 3)
            self.positions.data[i] = x, y


# Create the text
s = "**Lorem ipsum** dolor sit amet, *consectetur adipiscing elit*, sed do eiusmod tempor ..."
text1 = gfx.Text(
    SnakeTextGeometry(markdown=s, font_size=10),
    gfx.TextMaterial(color="#fff"),
)
scene.add(text1)

# Camera and controller
camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.show_object(text1)
controller = gfx.OrbitController(camera, register_events=renderer)

# Put the scene in as box, with lights, for visual appeal.
box = gfx.Mesh(
    gfx.box_geometry(1000, 1000, 1000),
    gfx.MeshPhongMaterial(pick_write=True),
)
scene.add(box)
scene.add(gfx.AmbientLight())
light = gfx.DirectionalLight(0.2)
light.local.position = (0, 0, 1)
scene.add(camera.add(light))


@renderer.add_event_handler("pointer_down")
def handle_event(event):
    info = event.pick_info
    for key, val in info.items():
        print(key, "=", val)


def animate():
    text1.geometry.apply_layout()
    gfx.render_with_logo(renderer, scene, camera)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
