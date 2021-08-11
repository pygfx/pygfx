"""
Example showing a single geometric cube.
"""

import imageio
import pygfx as gfx

from PySide6 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
canvas._target_fps = 1000
renderer = gfx.renderers.WgpuRenderer(canvas, show_fps=True)
scene = gfx.Scene()

im = imageio.imread("imageio:bricks.jpg")
tex = gfx.Texture(im, dim=2).get_view(filter="linear", address_mode="repeat")

geometry = gfx.BoxGeometry(200, 200, 200)
geometry.texcoords.data[:] *= 2  # smaller bricks
material = gfx.MeshPhongMaterial(map=tex, color=(1, 0, 0, 0.2))
cube = gfx.Mesh(geometry, material)
scene.add(cube)


camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 400


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    cube.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec()
