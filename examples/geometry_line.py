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

material = vv.LineStripMaterial()
line = vv.Line(geometry, material)
scene.add(line)

camera = vv.ScreenCoordsCamera()


def animate():
    # would prefer to do this in a resize event only
    psize = canvas.get_physical_size()
    camera.set_viewport_size(*psize)
    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    renderer_svg.render(scene, camera)
    canvas.draw_frame = animate
    app.exec_()
