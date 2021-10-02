"""
Example demonstrating different colormap dimensions on a mesh.
"""

import pygfx as gfx
import numpy as np
import imageio

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])

canvas = WgpuCanvas(size=(900, 400))
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

geometry = gfx.TorusKnotGeometry(1, 0.3, 128, 32)

camera = gfx.OrthographicCamera(16, 3)


def create_object(texcoords, tex, xpos):
    geometry = gfx.TorusKnotGeometry(1, 0.3, 128, 32)
    geometry.texcoords = gfx.Buffer(texcoords, usage="vertex|storage")
    material = gfx.MeshPhongMaterial(map=tex, clim=(-0.05, 1))
    obj = gfx.Mesh(geometry, material)
    obj.position.x = xpos
    scene.add(obj)


# === 1D colormap
#
# For the 1D texcoords we use the first dimension of the default
# texcoords, which runs along the tube. The 1D colormap runs from yellow
# to green to red and back to yellow.

texcoords1 = geometry.texcoords.data[:, 0].copy()

cmap1 = np.array([(1, 1, 0, 1), (0, 1, 0, 1), (1, 0, 0, 1), (1, 1, 0, 1)], np.float32)
tex1 = gfx.Texture(cmap1, dim=1).get_view(filter="linear")

create_object(texcoords1, tex1, -6)

# === 2D colormap
#
# For the 2D texcoords we use the default texcoords, but multiply the
# first dimension so that the texture that we apply is repeated. For
# the 2D colormap we use an image texture.

texcoords2 = geometry.texcoords.data.copy()
texcoords2[:, 0] *= 10

cmap2 = imageio.imread("imageio:chelsea.png").astype(np.float32) / 255
tex2 = gfx.Texture(cmap2, dim=2).get_view(address_mode="repeat")

create_object(texcoords2, tex2, -2)


# === 3D colormap
#
# For the 3D texcoords we use (a scaled version of) the positions. For
# the colormap we use a volume (a 3D image). In effect, the edge of the
# mesh gets a color that corresponds to the value of the volume at that
# position. This can be seen as a specific (maybe somewhat odd) type
# of volume rendering.

texcoords3 = geometry.positions.data * 0.5 + 0.5

cmap3 = imageio.volread("imageio:stent.npz")
tex3 = gfx.Texture(cmap3, dim=3).get_view()

create_object(texcoords3, tex3, +2)


# === Per vertex coloring
#
# To specify a color for each vertex, we also use 3D texture coordinates.
# In this case we use the normals (a normal in the x direction would be red).
# To make this work, we use a 3D colormap that represents an RGB color cube.

texcoords4 = geometry.normals.data * 0.4 + 0.5

cmap4 = np.array(
    [
        [
            [(0, 0, 0, 1), (1, 0, 0, 1)],
            [(0, 1, 0, 1), (1, 1, 0, 1)],
        ],
        [
            [(0, 0, 1, 1), (1, 0, 1, 1)],
            [(0, 1, 1, 1), (1, 1, 1, 1)],
        ],
    ],
    np.float32,
)
tex4 = gfx.Texture(cmap4, dim=3).get_view()

create_object(texcoords4, tex4, +6)


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.0071, 0.01))
    for obj in scene.children:
        obj.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
