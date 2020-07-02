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
images = []
# base_url = "https://raw.githubusercontent.com/imageio/imageio-binaries/master/images/"
base_url = "C:/dev/pylib/imageio-binaries/images/"
for suffix in ("negx", "negy", "negz", "posx", "posy", "posz"):
    im = imageio.imread(base_url + "meadow_" + suffix + ".jpg")
    im = np.concatenate([im, np.ones(im.shape[:2] + (1,), dtype=im.dtype)], 2)
    im = im[::3, ::3, :]  # todo: ? prevent TooManyObjects error
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

geometry = gfx.BoxGeometry(200, 200, 200)
material = gfx.MeshBasicMaterial(map=view, clim=(0.2, 0.8))
cube = gfx.Mesh(geometry, material)
scene.add(cube)

camera = gfx.PerspectiveCamera(70)
camera.position.z = 0


def animate():
    # cube.rotation.x += 0.005
    # cube.rotation.y += 0.01
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.0005, 0.001))
    cube.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
