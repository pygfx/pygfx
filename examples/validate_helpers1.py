"""
Example showing the axes helper.

* The axes must be centered in the middle.
* The orange axes (x) must be to the right.
* The yellow axes (y) must be to the top.
* The blue axes (z) is not visible.
"""

import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)

scene = gfx.Scene()
scene.add(gfx.AxesHelper(length=40))
camera = gfx.OrthographicCamera(100, 100)


if __name__ == "__main__":
    print(__doc__)
    canvas.request_draw(lambda: renderer.render(scene, camera))
    app.exec_()
