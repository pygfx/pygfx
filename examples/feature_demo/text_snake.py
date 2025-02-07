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


# Create the text
s = "**Lorem ipsum** dolor sit amet, *consectetur adipiscing elit*, sed do eiusmod tempor ..."
text1 = gfx.Text(
    gfx.TextGeometry(markdown=s, font_size=10),
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
    yy = text1.geometry.glyph_positions.data[:, 1]
    tt = np.linspace(0, 10, len(yy))
    yy += np.sin(tt + 2 * perf_counter())
    text1.geometry.glyph_positions.update_full()

    renderer.render(scene, camera)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
