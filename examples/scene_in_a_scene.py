"""
Example of a scene rendered to a texture, which is shown in another scene.

Inception style. Since both the outer and inner scenes are lit, this
may look a bit weird. The faces of the outer cube are like "screens"
on which the subscene is displayed.

We user the offscreen canvas as a placeholder for the
renderer. We don't call canvas1.draw() (which would return a numpy array
and thus involve a memcopy from GPU to CPU), but control the renderer
directly.
"""

import numpy as np
import imageio
import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])


# Then create the actual scene, in the visible canvas

canvas2 = WgpuCanvas()

renderer2 = gfx.renderers.WgpuRenderer(canvas2)
scene2 = gfx.Scene()

texture = gfx.Texture(
    dim=2,
    size=(200, 200, 1),
    format="rgba8unorm",
    usage="TEXTURE_BINDING|RENDER_ATTACHMENT",
)
texture_view = texture.get_view(filter="linear", address_mode="repeat")

geometry2 = gfx.BoxGeometry(200, 200, 200)
# todo: the clim being 0..255 feels weird here, maybe review that
material2 = gfx.MeshPhongMaterial(map=texture_view, clim=(0, 255))
cube2 = gfx.Mesh(geometry2, material2)
scene2.add(cube2)

camera2 = gfx.PerspectiveCamera(70, 16 / 9)
camera2.position.z = 400


# First create the subscene, in an offscreen canvas

renderer1 = gfx.renderers.WgpuRenderer(texture)
scene1 = gfx.Scene()

background1 = gfx.Background(gfx.BackgroundMaterial((0, 0.5, 0, 1)))
scene1.add(background1)

im = imageio.imread("imageio:bricks.jpg").astype(np.float32) / 255
tex = gfx.Texture(im, dim=2).get_view(filter="linear", address_mode="repeat")
geometry1 = gfx.BoxGeometry(200, 200, 200)
material1 = gfx.MeshPhongMaterial(map=tex, color=(1, 1, 0, 1.0), clim=(0, 1))
cube1 = gfx.Mesh(geometry1, material1)
scene1.add(cube1)

camera1 = gfx.PerspectiveCamera(70, 16 / 9)
camera1.position.z = 300


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    cube1.rotation.multiply(rot)
    cube2.rotation.multiply(rot)

    with renderer1:
        renderer1.render(scene1, camera1)
    with renderer2:
        renderer2.render(scene2, camera2)

    canvas2.request_draw()


if __name__ == "__main__":
    canvas2.request_draw(animate)
    app.exec_()
