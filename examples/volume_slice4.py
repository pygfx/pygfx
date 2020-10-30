"""
Render slices through a volume, by creating a 3D texture, and viewing
it with a VolumeSliceMaterial. Easy because we can just define the view plane.
"""

import imageio
import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


class WgpuCanvasWithScroll(WgpuCanvas):
    def wheelEvent(self, event):  # noqa: N802
        degrees = event.angleDelta().y() / 8
        scroll(degrees)

    def mousePressEvent(self, event):  # noqa: N802
        # Print the voxel coordinate being clicked
        xy = event.x(), event.y()
        info = renderer.get_info_at(xy)
        if "xyz" in info:
            print(info["xyz"])


app = QtWidgets.QApplication([])
canvas = WgpuCanvasWithScroll()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

voldata = imageio.volread("imageio:stent.npz")
nslices = voldata.shape[0]
index = nslices // 2

tex = gfx.Texture(voldata, dim=3, usage="sampled")
vol = gfx.Volume(
    tex.size, gfx.VolumeSliceMaterial(map=tex, clim=(0, 255), plane=(0, 0, -1, index))
)
scene.add(vol)

camera = gfx.OrthographicCamera(128, 128)
camera.position.set(64, 64, 128)


def scroll(degrees):
    global index
    index = index + degrees / 30
    index = max(0, min(nslices - 1, index))
    vol.material.plane = 0, 0, -1, index
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    app.exec_()
