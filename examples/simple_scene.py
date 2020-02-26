from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

import visvis2 as vv

app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = vv.WgpuSurfaceRenderer(canvas)

scene = vv.Scene()


class Triangle(vv.WorldObject):
    def __init__(self, material):
        super().__init__()
        self.material = material


t1 = Triangle(vv.TriangleMaterial())
scene.add(t1)

for i in range(2):
    scene.add(Triangle(vv.TriangleMaterial()))


camera = vv.NDCCamera()  # This material does not use the camera anyway :P


def animate():
    width, height, ratio = canvas.get_size_and_pixel_ratio()
    camera.update_viewport_size(width, height)

    renderer.render(scene, camera)


if __name__ == "__main__":
    canvas.draw_frame = animate
    app.exec_()
    canvas.closeEvent = lambda *args: app.quit()
