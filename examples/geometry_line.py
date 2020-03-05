import random

import visvis2 as vv

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = vv.WgpuRenderer(canvas)
renderer_svg = vv.SvgRenderer(640, 480, "~/line.svg")

scene = vv.Scene()

positions = [[10 + i * 20, 100 + random.uniform(0, 40), 0, 1] for i in range(20)]
geometry = vv.Geometry(positions=positions)
geometry.positions.set_mapped(True)

material = vv.LineStripMaterial()
line = vv.Line(geometry, material)
scene.add(line)

camera = vv.ScreenCoordsCamera()


def animate():
    # would prefer to do this in a resize event only
    width, height, ratio = canvas.get_size_and_pixel_ratio()
    camera.set_viewport_size(width, height)
    renderer.render(scene, camera)
    canvas.update()


if __name__ == "__main__":
    renderer_svg.render(scene, camera)
    canvas.draw_frame = animate
    app.exec_()
