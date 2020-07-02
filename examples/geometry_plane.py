"""
Use a plane geometry to show a texture, which is continuously updated to show video.
"""

import time
import imageio
import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

reader = imageio.get_reader("imageio:cockatoo.mp4")
im = reader.get_next_data()[:, :, 1]

tex = gfx.Texture(im, dim=2, usage="sampled")

geometry = gfx.PlaneGeometry(200, 200, 12, 12)
material = gfx.MeshBasicMaterial(map=tex.get_view(filter="linear"), clim=(0, 255))
plane = gfx.Mesh(geometry, material)
plane.scale.y = -1
scene.add(plane)

camera = gfx.PerspectiveCamera(70)
camera.position.z = 200
camera.scale.y = -1


t = time.time()


def animate():
    global t

    if time.time() - t > 0.05:
        # Read next frame, rewind if we reach the end
        t = time.time()
        try:
            tex.data[:] = reader.get_next_data()[:, :, 1]
        except IndexError:
            reader.set_image_index(0)
        else:
            tex.update_range((0, 0, 0), tex.size)
            material.dirty = 1

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.draw_frame = animate
    app.exec_()
