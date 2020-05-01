"""
Use a plane geometry to show a texture, which is continuously updated to show video.
"""

import time
import imageio
import visvis2 as vv

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = vv.renderers.WgpuRenderer(canvas)
scene = vv.Scene()

reader = imageio.get_reader("imageio:cockatoo.mp4")
im = reader.get_next_data()[:, :, 1]

tex = vv.Texture(im, dim=2, usage="sampled")

geometry = vv.PlaneGeometry(200, 200, 12, 12)
material = vv.MeshBasicMaterial(map=tex.get_view(filter="linear"), clim=(0, 255))
plane = vv.Mesh(geometry, material)
plane.scale.y = -1
scene.add(plane)

fov, aspect, near, far = 70, -16 / 9, 1, 1000
camera = vv.PerspectiveCamera(fov, aspect, near, far)
camera.position.z = 200


t = time.time()


def animate():
    global t

    # would prefer to do this in a resize event only
    physical_size = canvas.get_physical_size()
    camera.set_viewport_size(*physical_size)

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

    # actually render the scene
    renderer.render(scene, camera)

    # Request new frame
    canvas.request_draw()


if __name__ == "__main__":
    canvas.draw_frame = animate
    app.exec_()
