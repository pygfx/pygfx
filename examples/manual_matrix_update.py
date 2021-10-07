"""
Example showing transform control flow without matrix auto updating.
"""

import imageio
import pygfx as gfx

from PySide6 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

im = imageio.imread("imageio:chelsea.png")
tex = gfx.Texture(im, dim=2).get_view(filter="linear")

material = gfx.MeshBasicMaterial(map=tex)
geometry = gfx.BoxGeometry(100, 100, 100)
cubes = [gfx.Mesh(geometry, material) for i in range(8)]
for i, cube in enumerate(cubes):
    cube.matrix_auto_update = False
    cube.matrix = gfx.linalg.Matrix4().set_position_xyz(350 - i * 100, 0, 0)
    scene.add(cube)

background = gfx.Background(gfx.BackgroundMaterial((0, 1, 0, 1), (0, 1, 1, 1)))
scene.add(background)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.matrix_auto_update = False
camera.matrix = gfx.linalg.Matrix4().set_position_xyz(0, 0, 500)


def animate():
    for i, cube in enumerate(cubes):
        pos = gfx.linalg.Matrix4().set_position_xyz(350 - i * 100, 0, 0)
        rot = gfx.linalg.Matrix4().extract_rotation(cube.matrix)
        rot.premultiply(
            gfx.linalg.Matrix4().make_rotation_from_euler(
                gfx.linalg.Euler(0.01 * i, 0.02 * i)
            )
        )
        rot.premultiply(pos)
        cube.matrix = rot

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
