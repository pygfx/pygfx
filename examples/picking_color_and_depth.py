"""
Example showing picking the color and depth from the scene. This info
is always available, regardless of the object being clicked. See the
other examples for using picking info specific to certain
objects/materials.
"""

import numpy as np
import imageio
import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


class PickingWgpuCanvas(WgpuCanvas):
    def mousePressEvent(self, event):  # noqa: N802
        # Get a dict with info about the clicked location
        xy = event.x(), event.y()
        info = renderer.get_info_at(xy)
        # Add more info
        pos = gfx.linalg.Vector3(*info["ndc"])
        pos.apply_matrix4(camera.projection_matrix_inverse)
        pos.apply_matrix4(camera.matrix_world)
        info["position"] = tuple(pos.to_array())
        info["distance_to_camera"] = pos.distance_to(camera.position)
        # Show
        for key, val in info.items():
            print(key, "=", val)


app = QtWidgets.QApplication([])
canvas = PickingWgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

im = imageio.imread("imageio:astronaut.png").astype(np.float32) / 255
tex = gfx.Texture(im, dim=2, usage="sampled").get_view(
    filter="linear", address_mode="repeat"
)

geometry = gfx.BoxGeometry(200, 200, 200)
material = gfx.MeshBasicMaterial(map=tex)
cube = gfx.Mesh(geometry, material)
scene.add(cube)


# camera = gfx.OrthographicCamera(300, 300)
camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 400


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    cube.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
