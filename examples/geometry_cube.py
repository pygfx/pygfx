import visvis2 as vv

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

app = QtWidgets.QApplication([])


canvas = WgpuCanvas()
size = [800, 600]

scene = vv.Scene()

geometry = vv.BoxGeometry(200, 200, 200)
material = vv.MeshBasicMaterial()
cube = vv.Mesh(geometry, material)
scene.add(cube)

fov, aspect, near, far = 70, 16/9, 1, 1000
camera = vv.PerspectiveCamera(fov, aspect, near, far)
camera.position.z = 400

renderer = vv.SurfaceWgpuRenderer()


def animate():
    # would prefer to do this in a resize event only
    width, height, ratio = canvas.getSizeAndPixelRatio()
    camera.aspect = width / height
    camera.updateProjectionMatrix()
    renderer.setSize(width, height)
    renderer.setDevicePixelRatio(ratio)

    # actually render the scene
    renderer.render(scene, camera)


if __name__ == "__main__":
    canvas.drawFrame = animate
    app.exec_()
