"""
Render slices through a volume, by creating a 3D texture, and sample it in the shader.
Simple, fast and subpixel!
"""

import imageio
import numpy as np
import visvis2 as vv

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


class WgpuCanvasWithScroll(WgpuCanvas):
    def wheelEvent(self, event):  # noqa: N802
        degrees = event.angleDelta().y() / 8
        scroll(degrees)


app = QtWidgets.QApplication([])

canvas = WgpuCanvasWithScroll()
renderer = vv.renderers.WgpuRenderer(canvas)
scene = vv.Scene()

vol = imageio.volread("imageio:stent.npz")[::2, ::2, ::2]
nslices = vol.shape[0]
index = nslices // 2

tex = vv.Texture(vol, dim=3, usage="sampled")
view = tex.get_view(filter="linear")

geometry = vv.PlaneGeometry(200, 200, 1, 1)
texcoords = np.hstack(
    [geometry.texcoords.data, np.ones((4, 1), np.float32) * nslices / 2]
)
geometry.texcoords = vv.Buffer(texcoords, usage="vertex|storage")

material = vv.MeshVolumeSliceMaterial(map=view, clim=(0, 255))
plane = vv.Mesh(geometry, material)
plane.scale.y = -1
scene.add(plane)

fov, aspect, near, far = 70, -16 / 9, 1, 1000
camera = vv.PerspectiveCamera(fov, aspect, near, far)
camera.position.z = 200


def scroll(degrees):
    global index
    index = index + degrees / 30
    index = max(0, min(nslices - 1, index))
    geometry.texcoords.data[:, 2] = index / nslices
    geometry.texcoords.update_range(0, geometry.texcoords.nitems)
    material.dirty = 1  # todo: we should not have to mark the *material* dirty!
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
