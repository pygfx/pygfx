"""
Volume Slice 2
==============

Render slices through a volume, by creating a 3D texture, and sampling onto
a plane geometry. Simple, fast and subpixel!
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import imageio.v3 as iio
import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

vol = iio.imread("imageio:stent.npz").astype("float32") / 2000
nslices = vol.shape[0]
index = nslices // 2

tex = gfx.Texture(vol, dim=3)

geometry = gfx.plane_geometry(200, 200, 1, 1)
texcoords = np.hstack([geometry.texcoords.data, np.ones((4, 1), np.float32) * 0.5])
geometry.texcoords = gfx.Buffer(texcoords)

material = gfx.MeshBasicMaterial(map=tex)
plane = gfx.Mesh(geometry, material)
scene.add(plane)

camera = gfx.OrthographicCamera(200, 200)


@renderer.add_event_handler("wheel")
def handle_event(event):
    global index
    index = index + event.dy / 90
    index = max(0, min(nslices - 1, index))
    geometry.texcoords.data[:, 2] = index / nslices
    geometry.texcoords.update_full()
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
