"""
Text multiple grid
==================

This example demonstrate how to use MultiText to show a grid of text blocks.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from wgpu.gui.auto import WgpuCanvas, run

import pygfx as gfx
import numpy as np


scene = gfx.Scene()
scene.add(gfx.Background.from_color("#000"))

# Create text object
text = gfx.MultiText(
    anchor="top-left",
    anchor_offset=0.5,
    screen_space=False,
    font_size=3,
    material=gfx.TextMaterial(color="#fff", outline_thickness=0.2),
)
scene.add(text)


def pseudo_random(*seeds):
    seeds = np.array(seeds, float) + 317
    return np.sin(np.prod(seeds)) / 2 + 1


# Create a bunch of objects, and a textBlock for each
count = 0
for iy in range(10):
    for ix in range(10):
        count += 1
        x = ix * 10
        y = 100 - iy * 10

        rgba = pseudo_random(x, y, 1), pseudo_random(x, y, 2), pseudo_random(x, y, 3), 1
        ob = gfx.Mesh(
            gfx.plane_geometry(8, 8),
            gfx.MeshBasicMaterial(color=rgba),
        )
        ob.local.x = x + 4
        ob.local.y = y - 4
        scene.add(ob)

        block = text.create_text_block()
        block.set_text(str(count))
        block.set_position(x, y)


camera = gfx.OrthographicCamera(100, 100)
camera.local.position = 51, 51, 0

renderer = gfx.renderers.WgpuRenderer(WgpuCanvas(size=(800, 600)))
controller = gfx.PanZoomController(camera, register_events=renderer)

renderer.request_draw(lambda: renderer.render(scene, camera))


if __name__ == "__main__":
    run()
