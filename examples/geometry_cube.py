import visvis2 as vv

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = vv.WgpuSurfaceRenderer(canvas)

scene = vv.Scene()

geometry = vv.BoxGeometry(200, 200, 200)
material = vv.MeshBasicMaterial()
cube = vv.Mesh(geometry, material)
scene.add(cube)

fov, aspect, near, far = 70, 16 / 9, 1, 1000
camera = vv.PerspectiveCamera(fov, aspect, near, far)
camera.position.z = 400


def animate():
    # would prefer to do this in a resize event only
    width, height, ratio = canvas.get_size_and_pixel_ratio()
    camera.update_viewport_size(width, height)

    # cube.rotation.x += 0.005
    # cube.rotation.y += 0.01
    rot = vv.linalg.Quaternion().set_from_euler(vv.linalg.Euler(0.005, 0.01))
    cube.rotation.multiply(rot)

    # actually render the scene
    renderer.render(scene, camera)

    # Request new frame
    canvas.update()


if __name__ == "__main__":
    canvas.draw_frame = animate
    app.exec_()
