"""
Example showing multiple rotating cubes. This also tests the depth buffer.
"""

import numpy as np
import imageio
import visvis2 as vv

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = vv.renderers.WgpuRenderer(canvas)
scene = vv.Scene()

im = imageio.imread("imageio:chelsea.png")[:250, :250, :].astype(np.float32) / 255
im = np.concatenate([im, np.ones(im.shape[:2] + (1,), dtype=im.dtype)], 2)
tex = vv.Texture(im, dim=2, usage="sampled").get_view(filter="linear")

material = vv.MeshBasicMaterial(map=tex, clim=(0.2, 0.8))
geometry = vv.BoxGeometry(100, 100, 100)
cubes = [vv.Mesh(geometry, material) for i in range(8)]
for i, cube in enumerate(cubes):
    cube.position.set(350 - i * 100, 0, 0)
    scene.add(cube)

fov, aspect, near, far = 70, 16 / 9, 1, 1000
camera = vv.PerspectiveCamera(fov, aspect, near, far)
camera.position.z = 400


def animate():
    # would prefer to do this in a resize event only
    physical_size = canvas.get_physical_size()
    camera.set_viewport_size(*physical_size)

    for i, cube in enumerate(cubes):
        rot = vv.linalg.Quaternion().set_from_euler(
            vv.linalg.Euler(0.0005 * i, 0.001 * i)
        )
        cube.rotation.multiply(rot)

    # actually render the scene
    renderer.render(scene, camera)

    # Request new frame
    canvas.request_draw()


if __name__ == "__main__":
    canvas.draw_frame = animate
    app.exec_()
