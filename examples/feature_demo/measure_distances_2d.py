"""
Measure distances in 2D
=======================

Example to do measurements in a 2D image. Use LMB and RMB to place the
end-points.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()


im = iio.imread("imageio:astronaut.png")
image = gfx.Image(
    gfx.Geometry(grid=gfx.Texture(im, dim=2)),
    gfx.ImageBasicMaterial(clim=(0, 255), pick_write=True),
)
scene.add(image)

ruler = gfx.Ruler(ticks_at_end_points=True)
ruler.local.z = 0.1  # move on top of the image
scene.add(ruler)

camera = gfx.OrthographicCamera(512, 512)
camera.local.position = (256, 256, 0)
camera.local.scale_y = -1


@image.add_event_handler("click")
def event_handler(event):
    pos = np.array([*event.pick_info["index"], 0])
    if event.button == 1:
        ruler.start_pos = pos
    elif event.button == 2:
        ruler.end_pos = pos
    renderer.request_draw()


def animate():
    ruler.update(camera, canvas.get_logical_size())
    renderer.render(scene, camera)


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
