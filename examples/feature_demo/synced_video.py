"""
Synced Video Rendering
======================

Example demonstrating synced video rendering
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas(size=(512 * 3 + 100, 400), max_fps=999)
renderer = gfx.renderers.WgpuRenderer(canvas, show_fps=True)
scene = gfx.Scene()
camera = gfx.OrthographicCamera(1800, 550)
camera.local.y = 256
camera.local.scale_y = -1
camera.local.x = 1736 / 2

colormap1 = gfx.cm.plasma


# Just create a 512x512 random image
def create_random_image():
    rand_img = np.random.rand(512, 512).astype(np.float32) * 255

    return gfx.Image(
        gfx.Geometry(grid=gfx.Texture(rand_img, dim=2)),
        gfx.ImageBasicMaterial(clim=(0, 255), map=colormap1),
    )


images = list()
img_pos = 0


for _i in range(3):
    images.append(create_random_image())
    images[-1].local.x = img_pos
    img_pos += 50 + 512
    scene.add(images[-1])


def animate():
    # update with new random image
    for img in images:
        img.geometry.grid.data[:] = np.random.rand(512, 512).astype(np.float32) * 255
        img.geometry.grid.update_range((0, 0, 0), img.geometry.grid.size)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
