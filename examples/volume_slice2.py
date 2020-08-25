"""
Render slices through a volume, by creating a 3D texture, with 2D views.
Simple and relatively fast, but no subslices.
"""

# todo: not working ATM, a Rust assertion fails in _update_texture(),
# let's wait til the next release of wgpu-native and try again

import imageio
import pygfx as gfx

from PyQt5 import QtWidgets
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

tex_size = tuple(reversed(vol.shape))
tex = gfx.Texture(vol, dim=2, size=tex_size, usage="sampled")
view = tex.get_view(filter="linear", view_dim="2d", layer_range=range(index, index + 1))

geometry = gfx.PlaneGeometry(200, 200, 12, 12)
material = gfx.MeshBasicMaterial(map=view, clim=(0, 255))
plane = gfx.Mesh(geometry, material)
plane.scale.y = -1
scene.add(plane)

camera = gfx.OrthographicCamera(200, 200)


def scroll(degrees):
    global index
    index = index + int(degrees / 15)
    index = max(0, min(nslices - 1, index))
    view = tex.get_view(
        filter="linear", view_dim="2d", layer_range=range(index, index + 1)
    )
    material.map = view
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    app.exec_()
