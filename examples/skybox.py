"""
Example showing a single geometric cube.

Inspired by https://github.com/gfx-rs/wgpu-rs/blob/master/examples/skybox/main.rs
"""

# todo: not working ATM, a Rust assertion fails in _update_texture(),
# let's wait til the next release of wgpu-native and try again

import numpy as np
import imageio
import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

# Read images
# The order of the images here is determined by how the GPU samples cube textures.
images = []
# base_url = "https://raw.githubusercontent.com/imageio/imageio-binaries/master/images/"
base_url = "C:/dev/pylib/imageio-binaries/images/"
for suffix in ("posx", "negx", "posy", "negy", "posz", "negz"):
    im = imageio.imread(base_url + "meadow_" + suffix + ".jpg")
    im = np.concatenate([im, 255 * np.ones(im.shape[:2] + (1,), dtype=im.dtype)], 2)
    images.append(im)

# Turn it into a 3D image (a 4d nd array)
cubemap_image = np.concatenate(images, 0).reshape(-1, *images[0].shape)

app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# Create cubemap texture
tex_size = images[0].shape[0], images[0].shape[1], 6
tex = gfx.Texture(cubemap_image, dim=2, size=tex_size, usage="sampled")
view = tex.get_view(view_dim="cube", layer_range=range(6))

# And the background image with the cube texture
background = gfx.Background(gfx.BackgroundImageMaterial(map=view))
scene.add(background)

# Let's add some cubes to make the scene less boring
cubes = []
for pos in (-600, 0, -600), (-600, 0, +600), (+600, 0, -600), (+600, 0, +600):
    clr = (0.5, 0.6, 0.0, 1.0)
    cube = gfx.Mesh(gfx.BoxGeometry(200, 200, 200), gfx.MeshBasicMaterial(color=clr))
    cube.position.from_array(pos)
    cubes.append(cube)
    scene.add(cube)


camera = gfx.PerspectiveCamera(70)
camera.position.z = 0


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.0005, 0.001))
    for cube in cubes:
        cube.rotation.multiply(rot)

    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0, 0.0005))
    camera.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
