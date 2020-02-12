# NOTE: CURRENTLY BROKEN - WIP

import visvis2 as vv

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = vv.WgpuSurfaceRenderer(canvas)

scene = vv.Scene()

geometry = vv.BoxGeometry(1, 1, 1)
material = vv.MeshBasicMaterial()
cube = vv.Mesh(geometry, material)
scene.add(cube)

# todo: note that the camera is not yet actually used :P
fov, aspect, near, far = 70, 16 / 9, 1, 1000
camera = vv.PerspectiveCamera(fov, aspect, near, far)
camera.position.z = 400


def animate():
    # would prefer to do this in a resize event only
    width, height, ratio = canvas.get_size_and_pixel_ratio()
    camera.aspect = width / height
    camera.update_projection_matrix()

    # actually render the scene
    renderer.render(scene, camera)


if __name__ == "__main__":
    canvas.draw_frame = animate
    app.exec_()
