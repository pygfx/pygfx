"""
Volume Slice 1
==============

Render slices through a volume, by uploading to a 2D texture.
Simple and ... slow.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

vol = iio.imread("imageio:stent.npz").astype("float32") / 2000
nslices = vol.shape[0]
index = nslices // 2
im = vol[index].copy()

tex = gfx.Texture(im, dim=2)

geometry = gfx.plane_geometry(200, 200, 12, 12)
material = gfx.MeshBasicMaterial(map=tex)
plane = gfx.Mesh(geometry, material)
scene.add(plane)

camera = gfx.OrthographicCamera(200, 200)


@renderer.add_event_handler("wheel")
def handle_event(event):
    global index
    index = index + int(event.dy / 90)
    index = max(0, min(nslices - 1, index))
    im = vol[index]
    tex.data[:] = im
    tex.update_full()
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
