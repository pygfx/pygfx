"""
Example demonstrating colormaps in 4 modes: grayscale, gray+alpha, RGB, RGBA.
"""

import pygfx as gfx
import numpy as np

from PySide6 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])

canvas = WgpuCanvas(size=(900, 400))
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

geometry = gfx.TorusKnotGeometry(1, 0.3, 128, 32)
geometry.texcoords = gfx.Buffer(geometry.texcoords.data[:, 0])

camera = gfx.OrthographicCamera(16, 3)


def create_object(tex, xpos):
    material = gfx.MeshPhongMaterial(map=tex, clim=(-0.05, 1))
    obj = gfx.Mesh(geometry, material)
    obj.position.x = xpos
    scene.add(obj)


# === 1-channel colormap: grayscale

cmap1 = np.array([(1,), (0,), (0,), (1,)], np.float32)
tex1 = gfx.Texture(cmap1, dim=1).get_view(filter="linear")
create_object(tex1, -6)

# ==== 2-channel colormap: grayscale + alpha

cmap2 = np.array([(1, 1), (0, 1), (0, 0), (1, 0)], np.float32)
tex1 = gfx.Texture(cmap2, dim=1).get_view(filter="linear")
create_object(tex1, -2)

# === 3-channel colormap: RGB

cmap3 = np.array([(1, 1, 0), (0, 1, 0), (0, 1, 0), (1, 1, 0)], np.float32)
tex1 = gfx.Texture(cmap3, dim=1).get_view(filter="linear")
create_object(tex1, +2)

# === 4-channel colormap: RGBA

cmap4 = np.array([(1, 1, 0, 1), (0, 1, 0, 1), (0, 1, 0, 0), (1, 1, 0, 0)], np.float32)
tex1 = gfx.Texture(cmap4, dim=1).get_view(filter="linear")
create_object(tex1, +6)


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.0071, 0.01))
    for obj in scene.children:
        obj.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec()
