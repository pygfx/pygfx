# import asyncio

import visvis2 as vv

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

app = QtWidgets.QApplication([])

f = vv.Figure(canvas=WgpuCanvas())

v = vv.View()
f._views.append(v)  # todo: API?

t1 = vv.Triangle()
t2 = vv.Triangle()

v.scene.add(t1)  # todo: API?
v.scene.add(t2)
v.scene.add(vv.Triangle())
v.scene.add(vv.Triangle())


if __name__ == "__main__":
    # loop = asyncio.get_event_loop()
    # loop.run_forever()
    app.exec_()


# renderer = vv.WgpuRenderer(widget_ish_or_surface_maybe_non_qt_specific)
#
# camera = vv.Camera()
#
# scene = vv.Scene()
#
# controller = vv.QtPanZoomController(camera, widget)
#
# scene.add(t1)
#
# renderer.render(scene, camera)
