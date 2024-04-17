"""
Colormap Channels
=================

Example demonstrating colormaps in 4 modes: grayscale, gray+alpha, RGB, RGBA.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'off'

import numpy as np
import pygfx as gfx
import pylinalg as la


group = gfx.Group()


geometry = gfx.torus_knot_geometry(1, 0.3, 128, 32)
geometry.texcoords = gfx.Buffer(geometry.texcoords.data[:, 0])

camera = gfx.OrthographicCamera(16, 3)


def create_object(tex, xpos):
    material = gfx.MeshPhongMaterial(map=tex)
    obj = gfx.Mesh(geometry, material)
    obj.local.x = xpos
    group.add(obj)


# === 1-channel colormap: grayscale

cmap1 = np.array([(1,), (0,), (0,), (1,)], np.float32)
tex1 = gfx.Texture(cmap1, dim=1)
create_object(tex1, -6)

# ==== 2-channel colormap: grayscale + alpha

cmap2 = np.array([(1, 1), (0, 1), (0, 0), (1, 0)], np.float32)
tex1 = gfx.Texture(cmap2, dim=1)
create_object(tex1, -2)

# === 3-channel colormap: RGB

cmap3 = np.array([(1, 1, 0), (0, 1, 0), (0, 1, 0), (1, 1, 0)], np.float32)
tex1 = gfx.Texture(cmap3, dim=1)
create_object(tex1, +2)

# === 4-channel colormap: RGBA

cmap4 = np.array([(1, 1, 0, 1), (0, 1, 0, 1), (0, 1, 0, 0), (1, 1, 0, 0)], np.float32)
tex1 = gfx.Texture(cmap4, dim=1)
create_object(tex1, +6)


def animate():
    rot = la.quat_from_euler((0.0071, 0.01), order="XY")
    for obj in group.children:
        obj.local.rotation = la.quat_mul(rot, obj.local.rotation)


if __name__ == "__main__":
    disp = gfx.Display()
    disp.before_render = animate
    disp.show(group)
