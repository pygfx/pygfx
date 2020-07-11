"""
Example showing multiple rotating cubes. This also tests the depth buffer.
"""

import numpy as np
import imageio
import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


rotate_start = None
camera = gfx.PerspectiveCamera(70, 16 / 9)


class WgpuCanvasWithInputEvents(WgpuCanvas):
    def wheelEvent(self, event):  # noqa: N802
        # degrees = event.angleDelta().y() / 8
        pass

    def mousePressEvent(self, event):  # noqa: N802
        global rotate_start
        pos = event.windowPos()
        rotate_start = np.array([pos.x(), pos.y()])

    def mouseReleaseEvent(self, event):  # noqa: N802
        global rotate_start
        rotate_start = None

    def mouseMoveEvent(self, event):  # noqa: N802
        global rotate_start
        rotate_speed = 1.0
        pos = event.windowPos()
        rotate_end = np.array([pos.x(), pos.y()])
        delta = (rotate_end - rotate_start) * rotate_speed
        delta[1] *= -1  # flip y axis
        angles = 2 * np.pi * delta / self.height() * -1  # invert direction
        global camera
        rot = gfx.linalg.Quaternion().set_from_euler(
            gfx.linalg.Euler(angles[1], angles[0])
        )
        camera.rotation.multiply(rot)
        rotate_start = rotate_end

    def keyPressEvent(self, event):  # noqa: N802
        pass

    def keyReleaseEvent(self, event):  # noqa: N802
        pass


app = QtWidgets.QApplication([])

canvas = WgpuCanvasWithInputEvents()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

im = imageio.imread("imageio:chelsea.png").astype(np.float32) / 255
im = np.concatenate([im, np.ones(im.shape[:2] + (1,), dtype=im.dtype)], 2)
tex = gfx.Texture(im, dim=2, usage="sampled").get_view(filter="linear")

material = gfx.MeshBasicMaterial(map=tex, clim=(0.2, 0.8))
geometry = gfx.BoxGeometry(100, 100, 100)
cubes = [gfx.Mesh(geometry, material) for i in range(8)]
for i, cube in enumerate(cubes):
    cube.position.set(350 - i * 100, 0, 0)
    scene.add(cube)

background = gfx.Background(gfx.BackgroundMaterial((0, 1, 0, 1), (0, 1, 1, 1)))
scene.add(background)


camera.position.z = 500


def animate():
    for i, cube in enumerate(cubes):
        rot = gfx.linalg.Quaternion().set_from_euler(
            gfx.linalg.Euler(0.0005 * i, 0.001 * i)
        )
        cube.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
