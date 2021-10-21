"""
Render a volume and volume slices. You should see:
* On the left: a raycasted volume fit snugly inside a red box.
* On the right: three orthogonal slices right insidea - and through the middle - of the green box.
* The volume has its corners darker and very center brighter.
"""

import numpy as np
import pygfx as gfx

from PyQt6 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])
canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)

# Prepare a very small data volume. The data is integer and not uint8,
# so its not interpolated (a wgpu restriction). In this case this is intended.
voldata = np.ones((3, 3, 3), np.int16) * 200
voldata[1:-1, :, :] = 600
voldata[:, 1:-1, :] = 600
voldata[:, :, 1:-1] = 600
voldata[1, 1, 1] = 800

# Create a texture for it
tex = gfx.Texture(voldata, dim=3)

# Prepare two 3x3x3 boxes to indicate the proper position
box1 = gfx.Mesh(
    gfx.BoxGeometry(3.1, 3.1, 3.1),
    gfx.MeshBasicMaterial(color=(1, 0, 0, 1), wireframe=True, wireframe_thickness=2),
)
box2 = gfx.Mesh(
    gfx.BoxGeometry(3.1, 3.1, 3.1),
    gfx.MeshBasicMaterial(color=(0, 1, 0, 1), wireframe=True, wireframe_thickness=2),
)

# In scene1 we show a raycasted volume
scene1 = gfx.Scene()
vol = gfx.Volume(tex, gfx.VolumeRayMaterial(clim=(0, 2000)))
vol.position.set(-1, -1, -1)
scene1.add(vol, box1)

# In scene2 we show volume slices
scene2 = gfx.Scene()
slice1 = gfx.Volume(tex, gfx.VolumeSliceMaterial(clim=(0, 1000), plane=(0, 0, 1, 0)))
slice2 = gfx.Volume(tex, gfx.VolumeSliceMaterial(clim=(0, 1000), plane=(0, 1, 0, 0)))
slice3 = gfx.Volume(tex, gfx.VolumeSliceMaterial(clim=(0, 1000), plane=(1, 0, 0, 0)))
for slice in (slice1, slice2, slice3):
    slice.position.set(-1, -1, -1)
scene2.add(slice1, slice2, slice3, box2)

# Prepare a camera so we can see the result in 3D
camera = gfx.PerspectiveCamera(90, 16 / 9)
camera.position.z = 5
camera.position.y = 4
camera.position.x = 3


def animate():
    w, h = canvas.get_logical_size()
    renderer.render(
        scene1, camera, viewport=(0.0 * w, -0.4 * h, 0.7 * w, 1.4 * h), flush=False
    )
    renderer.render(
        scene2, camera, viewport=(0.5 * w, -0.4 * h, 0.7 * w, 1.4 * h), flush=False
    )
    renderer.flush()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec()
