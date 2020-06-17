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
view = tex.get_view(filter="linear", view_dim="2d", layer_range=(index, index + 1))

geometry = gfx.PlaneGeometry(200, 200, 12, 12)
material = gfx.MeshBasicMaterial(map=view, clim=(0, 255))
plane = gfx.Mesh(geometry, material)
plane.scale.y = -1
scene.add(plane)

fov, aspect, near, far = 70, -16 / 9, 1, 1000
camera = gfx.PerspectiveCamera(fov, aspect, near, far)
camera.position.z = 200


def scroll(degrees):
    global index
    index = index + int(degrees / 15)
    index = max(0, min(nslices - 1, index))
    view = tex.get_view(filter="linear", view_dim="2d", layer_range=(index, index + 1))
    material.map = view
    material.dirty = 1
    canvas.request_draw()


def animate():
    global t

    # would prefer to do this in a resize event only
    physical_size = canvas.get_physical_size()
    camera.set_viewport_size(*physical_size)

    # actually render the scene
    renderer.render(scene, camera)


if __name__ == "__main__":
    canvas.draw_frame = animate
    app.exec_()
