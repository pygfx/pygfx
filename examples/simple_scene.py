from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

import visvis2 as vv

app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = vv.WgpuSurfaceRenderer(canvas)

scene = vv.Scene()

t1 = vv.Mesh(vv.Geometry(range(3)), vv.TriangleMaterial())
scene.add(t1)

for i in range(2):
    scene.add(vv.Mesh(vv.Geometry(range(3)), vv.TriangleMaterial()))


camera = vv.NDCCamera()  # This material does not use the camera anyway :P


def animate():
    width, height, ratio = canvas.get_size_and_pixel_ratio()
    camera.update_viewport_size(width, height)

    renderer.render(scene, camera)


if __name__ == "__main__":
    canvas.draw_frame = animate
    app.exec_()
    canvas.closeEvent = lambda *args: app.quit()
