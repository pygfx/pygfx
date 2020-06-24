"""
Example showing multiple rotating cubes. This also tests the depth buffer.
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

im = imageio.imread("imageio:chelsea.png")[:250, :250, :].astype(np.float32) / 255
im = np.concatenate([im, np.ones(im.shape[:2] + (1,), dtype=im.dtype)], 2)
tex = gfx.Texture(im, dim=2, usage="sampled").get_view(filter="linear")

material = gfx.MeshBasicMaterial(map=tex, clim=(0.2, 0.8))
geometry = gfx.BoxGeometry(100, 100, 100)
cubes = [gfx.Mesh(geometry, material) for i in range(8)]
for i, cube in enumerate(cubes):
    cube.position.set(350 - i * 100, 0, 0)
    scene.add(cube)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 500


def animate():
    # would prefer to do this in a resize event only
    physical_size = canvas.get_physical_size()
    camera.set_viewport_size(*physical_size)

    for i, cube in enumerate(cubes):
        rot = gfx.linalg.Quaternion().set_from_euler(
            gfx.linalg.Euler(0.0005 * i, 0.001 * i)
        )
        cube.rotation.multiply(rot)

    # actually render the scene
    renderer.render(scene, camera)

    # Request new frame
    canvas.request_draw()


if __name__ == "__main__":
    canvas.draw_frame = animate
    app.exec_()
