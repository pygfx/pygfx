"""
Render slices through a volume, by creating a 3D texture, and sampling onto
a plane geometry. Simple, fast and subpixel!
"""

import imageio
import numpy as np
import pygfx as gfx

from PySide6 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


class WgpuCanvasWithScroll(WgpuCanvas):
    def wheelEvent(self, event):  # noqa: N802
        degrees = event.angleDelta().y() / 8
        scroll(degrees)


app = QtWidgets.QApplication([])
canvas = WgpuCanvasWithScroll()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

vol = imageio.volread("imageio:stent.npz")
nslices = vol.shape[0]
index = nslices // 2

tex = gfx.Texture(vol, dim=3)
view = tex.get_view(filter="linear")

geometry = gfx.PlaneGeometry(200, 200, 1, 1)
texcoords = np.hstack([geometry.texcoords.data, np.ones((4, 1), np.float32) * 0.5])
geometry.texcoords = gfx.Buffer(texcoords)

material = gfx.MeshBasicMaterial(map=view, clim=(0, 2000))
plane = gfx.Mesh(geometry, material)
plane.scale.y = -1
scene.add(plane)

camera = gfx.OrthographicCamera(200, 200)


def scroll(degrees):
    global index
    index = index + degrees / 30
    index = max(0, min(nslices - 1, index))
    geometry.texcoords.data[:, 2] = index / nslices
    geometry.texcoords.update_range(0, geometry.texcoords.nitems)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    app.exec_()
