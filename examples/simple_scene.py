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


camera = vv.PerspectiveCamera(45, 16 / 9, 0.1, 1000)


def animate():
    width, height, ratio = canvas.get_size_and_pixel_ratio()
    camera.aspect = width / height
    camera.update_projection_matrix()
    # Actually render the scene
    renderer.render(scene, camera)


if __name__ == "__main__":
    canvas.draw_frame = animate
    app.exec_()
    canvas.closeEvent = lambda *args: app.quit()
