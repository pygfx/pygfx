"""
Isosurface Volume Rendering
===========================

Render a 3D volume with isosurface rendering..
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

voldata = iio.imread("imageio:stent.npz").astype(np.float32)

geometry = gfx.Geometry(grid=voldata)
material = gfx.VolumeIsoMaterial(clim=(0, 2000), threshold=1000)

vol1 = gfx.Volume(geometry, material)
scene.add(vol1)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.show_object(scene, view_dir=(-1, -1, -1), up=(0, 0, 1))
controller = gfx.OrbitController(camera, register_events=renderer)


def animate():
    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
