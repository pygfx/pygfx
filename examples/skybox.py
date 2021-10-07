"""
Example with a skybox background.

Inspired by https://github.com/gfx-rs/wgpu-rs/blob/master/examples/skybox/main.rs
"""

import imageio
import pygfx as gfx

from PySide6 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

# Read the image
# The order of the images is already correct for GPU cubemap texture sampling
im = imageio.imread("imageio:meadow_cube.jpg")

# Turn it into a 3D image (a 4d nd array)
width = height = im.shape[1]
im.shape = -1, width, height, 3

app = QtWidgets.QApplication([])
canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# Create cubemap texture
tex_size = width, height, 6
tex = gfx.Texture(im, dim=2, size=tex_size)
view = tex.get_view(view_dim="cube", layer_range=range(6))

# And the background image with the cube texture
background = gfx.Background(gfx.BackgroundImageMaterial(map=view))
scene.add(background)

# Let's add some cubes to make the scene less boring
cubes = []
for pos in (-600, 0, -600), (-600, 0, +600), (+600, 0, -600), (+600, 0, +600):
    clr = (0.5, 0.6, 0.0, 1.0)
    cube = gfx.Mesh(gfx.BoxGeometry(200, 200, 200), gfx.MeshPhongMaterial(color=clr))
    cube.position.from_array(pos)
    cubes.append(cube)
    scene.add(cube)


camera = gfx.PerspectiveCamera(70)
camera.scale.y = -1
camera.position.z = 0


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.01, 0.02))
    for cube in cubes:
        cube.rotation.multiply(rot)

    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0, 0.005))
    camera.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
