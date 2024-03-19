"""
Volume and Volume Slice Rendering
=================================

Render a volume and volume slices. You should see:

* On the left: a raycasted volume fit snugly inside a red box.
* On the right: three orthogonal slices inside - and through the middle of - a green box.
* The volume has its corners darker and its very center is brighter.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)

# Prepare a very small data volume. The data is integer and not uint8,
# so its not interpolated (a wgpu restriction). In this case this is intended.
voldata = np.ones((3, 3, 3), np.int16) * 200
voldata[1:-1, :, :] = 600
voldata[:, 1:-1, :] = 600
voldata[:, :, 1:-1] = 600
voldata[1, 1, 1] = 800

# Create a texture, (wrapped in a geometry) for it
geo = gfx.Geometry(grid=gfx.Texture(voldata, dim=3))

# Prepare two 3x3x3 boxes to indicate the proper position
box1 = gfx.Mesh(
    gfx.box_geometry(3.1, 3.1, 3.1),
    gfx.MeshBasicMaterial(color=(1, 0, 0, 1), wireframe=True, wireframe_thickness=1),
)
box2 = gfx.Mesh(
    gfx.box_geometry(3.1, 3.1, 3.1),
    gfx.MeshBasicMaterial(color=(0, 1, 0, 1), wireframe=True, wireframe_thickness=1),
)

# In scene1 we show a raycasted volume
scene1 = gfx.Scene()
vol = gfx.Volume(geo, gfx.VolumeRayMaterial(clim=(0, 2000)))
vol.local.position = (-1, -1, -1)
scene1.add(vol, box1)

# In scene2 we show volume slices
scene2 = gfx.Scene()
slice1 = gfx.Volume(geo, gfx.VolumeSliceMaterial(clim=(0, 1000), plane=(0, 0, 1, 0)))
slice2 = gfx.Volume(geo, gfx.VolumeSliceMaterial(clim=(0, 1000), plane=(0, 1, 0, 0)))
slice3 = gfx.Volume(geo, gfx.VolumeSliceMaterial(clim=(0, 1000), plane=(1, 0, 0, 0)))
for slice in (slice1, slice2, slice3):
    slice.local.position = (-1, -1, -1)
scene2.add(slice1, slice2, slice3, box2)

# Prepare a camera so we can see the result in 3D
camera = gfx.PerspectiveCamera(90, 16 / 9, depth_range=(0.1, 2000))
camera.local.position = (3, 3, 1)
camera.look_at((0, 1, 0))


def animate():
    w, h = canvas.get_logical_size()
    renderer.render(
        scene1, camera, rect=(0.0 * w, 0.0 * h, 0.5 * w, 1.0 * h), flush=False
    )
    renderer.render(
        scene2, camera, rect=(0.5 * w, 0.0 * h, 0.5 * w, 1.0 * h), flush=False
    )
    renderer.flush()


canvas.request_draw(animate)

if __name__ == "__main__":
    run()
