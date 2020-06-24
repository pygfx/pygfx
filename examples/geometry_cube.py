"""
Example showing a single geometric cube.
"""

import numpy as np
import imageio
import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# todo: don't reduce image size, its needed now because a 1MB buffer size limit that has been fixed, but not yet in our build
im = imageio.imread("imageio:chelsea.png")[:250, :250, :].astype(np.float32) / 255
im = np.concatenate([im, np.ones(im.shape[:2] + (1,), dtype=im.dtype)], 2)
tex = gfx.Texture(im, dim=2, usage="sampled").get_view(filter="linear")

geometry = gfx.BoxGeometry(200, 200, 200)
material = gfx.MeshBasicMaterial(map=tex, clim=(0.2, 0.8))
cube = gfx.Mesh(geometry, material)
scene.add(cube)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 400


def animate():
    # would prefer to do this in a resize event only
    physical_size = canvas.get_physical_size()
    camera.set_viewport_size(*physical_size)

    # cube.rotation.x += 0.005
    # cube.rotation.y += 0.01
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.0005, 0.001))
    cube.rotation.multiply(rot)

    # actually render the scene
    renderer.render(scene, camera)

    # Request new frame
    canvas.request_draw()


if __name__ == "__main__":
    canvas.draw_frame = animate
    app.exec_()
