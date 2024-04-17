"""
Bounding Box Coordinates
========================

Render two volumes using different world transforms.
Prints the world bounding box of the scene which used
to trigger an Exception.
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
geometry_with_texture = gfx.Geometry(grid=gfx.Texture(data=voldata, dim=3))
material = gfx.VolumeRayMaterial(clim=(0, 2000))

vol1 = gfx.Volume(geometry, material)
vol2 = gfx.Volume(geometry_with_texture, material)
scene.add(vol1, vol2)

vol2.local.x = 150

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.show_object(scene, view_dir=(-1, -1, -1), up=(0, 0, 1))
controller = gfx.OrbitController(camera, register_events=renderer)

print("World bounding box:\n", scene.get_world_bounding_box())


def animate():
    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
