"""
Render slices through a volume, by uploading to a 2D texture.
Simple and ... slow.
"""

import imageio
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
im = vol[index].copy()

tex = gfx.Texture(im, dim=2)

geometry = gfx.PlaneGeometry(200, 200, 12, 12)
material = gfx.MeshBasicMaterial(map=tex.get_view(filter="linear"), clim=(0, 255))
plane = gfx.Mesh(geometry, material)
plane.scale.y = -1
scene.add(plane)

camera = gfx.OrthographicCamera(200, 200)


def scroll(degrees):
    global index
    index = index + int(degrees / 15)
    index = max(0, min(nslices - 1, index))
    im = vol[index]
    tex.data[:] = im
    tex.update_range((0, 0, 0), tex.size)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    app.exec_()
