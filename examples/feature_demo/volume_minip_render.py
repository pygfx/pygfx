"""
Minip Volume Rendering
======================

Render a 3D volume with minimum intensity projection rendering.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

voldata = np.ones((100, 100, 100), dtype=np.float32)
voldata[25:75, 25:75, 25:75] = 0.2

geometry = gfx.Geometry(grid=voldata)
material = gfx.VolumeMinipMaterial(clim=(0, 1), map=gfx.cm.viridis)

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
