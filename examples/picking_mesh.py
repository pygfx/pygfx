"""
Example showing picking a mesh. Showing two meshes that can be clicked
on. Upon clicking, the vertex closest to the pick location is moved.
"""

# todo: if we have per-vertex coloring, we can paint on the mesh instead :D

import numpy as np
import imageio
import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


class PickingWgpuCanvas(WgpuCanvas):
    def mousePressEvent(self, event):  # noqa: N802
        # Get a dict with info about the clicked location
        xy = event.x(), event.y()
        info = renderer.get_pick_info(xy)
        wobject = info["world_object"]
        # If a mesh was clicked ..
        if wobject and "face_index" in info:
            # Get what face was clicked
            face_index = info["face_index"]
            coords = info["face_coords"]
            # Select which of the three vertices was closest
            # Note that you can also select all vertices for this face,
            # or use the coords to select the closest edge.
            sub_index = np.argmax(coords)
            # Look up the vertex index
            vertex_index = int(wobject.geometry.index.data[face_index * 3 + sub_index])
            # Change the position of that vertex
            pos = wobject.geometry.positions.data[vertex_index]
            pos[:] *= 1.1
            wobject.geometry.positions.update_range(vertex_index, 1)


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
cube.position.x += 150
scene.add(cube)

torus = gfx.Mesh(gfx.TorusKnotGeometry(100, 20, 128, 32), gfx.MeshPhongMaterial())
torus.position.x -= 150
scene.add(torus)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 400


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    cube.rotation.multiply(rot)
    torus.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
