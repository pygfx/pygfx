"""
Show an image with points overlaid.
"""

import imageio
import pygfx as gfx
import numpy as np

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# %% add image

im = imageio.imread("imageio:astronaut.png")[::2, ::2, :]
im = np.concatenate([im, 255 * np.ones(im.shape[:2] + (1,), dtype=im.dtype)], 2)  # yuk!

tex = gfx.Texture(im, dim=2, usage="sampled")

geometry = gfx.PlaneGeometry(512, 512)
material = gfx.MeshBasicMaterial(map=tex.get_view(filter="linear"), clim=(0, 255))

plane = gfx.Mesh(geometry, material)
plane.position = gfx.linalg.Vector3(256, 256, 0)  # put corner at 0, 0

scene.add(plane)


# %% add points

xx = [182, 180, 161, 153, 191, 237, 293, 300, 272, 267, 254]
yy = [145, 131, 112, 59, 29, 14, 48, 91, 136, 137, 172]

geometry_p = gfx.Geometry(positions=[(x, y, 0, 1) for x, y in zip(xx, yy)])
material_p = gfx.PointsMaterial(color=(0, 1, 1, 1), size=10)
points = gfx.Points(geometry_p, material_p)
points.position.z = 1  # move points in front of the image
scene.add(points)


near, far = -400, 700
camera = gfx.OrthographicCamera(512, 512)
camera.position.set(256, 256, 0)


def animate():
    # would prefer to do this in a resize event only
    logical_size = canvas.get_logical_size()
    camera.set_viewport_size(*logical_size)
    # actually render the scene
    renderer.render(scene, camera)
    # Request new frame
    canvas.request_draw()


if __name__ == "__main__":
    canvas.draw_frame = animate
    app.exec_()
