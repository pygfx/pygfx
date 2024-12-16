"""
Mesh Colormaps
==============

Show meshes with 1D, 2D, and 3D colormaps, and per-vertex colors too.

* You should see four cylinders with block-pattern colors.
* The right-most cylinder is smoothly colored matching its normal.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la


canvas = WgpuCanvas(size=(900, 400))
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()


def get_geometry(**kwargs):
    geo = gfx.cylinder_geometry(
        height=2, radial_segments=32, height_segments=4, open_ended=False
    )
    for key, val in kwargs.items():
        setattr(geo, key, val)
    return geo


def WobjectClass(geometry, material):  # noqa
    return gfx.Mesh(geometry, material)
    # return gfx.Points(geometry, material)
    # return gfx.Line(geometry, material)


def MaterialClass(**kwargs):  # noqa
    map = kwargs.get("map")
    map = gfx.TextureMap(map, filter="nearest") if map is not None else None
    kwargs["map"] = map
    return gfx.MeshPhongMaterial(**kwargs)
    # return gfx.PointsMaterial(size=10, **kwargs)
    # return gfx.LineMaterial(thickness=5, **kwargs)


geometry = get_geometry()

camera = gfx.OrthographicCamera(16, 3, depth_range=(-1000, 1000))

reds = np.array([(0.1, 0, 0), (0.4, 0, 0), (0.6, 0, 0), (0.9, 0, 0)], np.float32)
greens = np.array([(0, 0.1, 0), (0, 0.4, 0), (0, 0.6, 0), (0, 0.9, 0)], np.float32)
blues = np.array([(0, 0, 0.1), (0, 0, 0.4), (0, 0, 0.6), (0, 0, 0.9)], np.float32)


# 1D

texcoords1 = geometry.texcoords.data[:, 1].copy()
cmap1 = reds

ob1 = WobjectClass(
    get_geometry(texcoords=gfx.Buffer(texcoords1)),
    MaterialClass(map=gfx.Texture(cmap1, dim=1)),
)
scene.add(ob1)
ob1.local.x = -6


# 2D

texcoords2 = geometry.texcoords.data.copy()
cmap2 = reds.reshape(1, -1, 3) + greens.reshape(-1, 1, 3)

ob2 = WobjectClass(
    get_geometry(texcoords=gfx.Buffer(texcoords2)),
    MaterialClass(map=gfx.Texture(cmap2, dim=2)),
)
scene.add(ob2)
ob2.local.x = -2


# 3D

texcoords3 = geometry.positions.data * 0.4 + 0.5
cmap3 = (
    reds.reshape(1, 1, -1, 3) + greens.reshape(1, -1, 1, 3) + blues.reshape(-1, 1, 1, 3)
)

ob3 = WobjectClass(
    get_geometry(texcoords=gfx.Buffer(texcoords3)),
    MaterialClass(map=gfx.Texture(cmap3, dim=3)),
)
scene.add(ob3)
ob3.local.x = +2


# Per vertex coloring

colors = geometry.normals.data * 0.4 + 0.5
colors = colors[:, :3]  # Colors can be Nx1, Nx2, Nx3, Nx4

ob4 = WobjectClass(
    get_geometry(colors=gfx.Buffer(colors)),
    MaterialClass(color_mode="vertex"),
)
scene.add(ob4)
ob4.local.x = +6


scene.add(gfx.AmbientLight(1, 0.2))

light = gfx.DirectionalLight(1, 2)
light.local.position = (0, 0, 1)
scene.add(light)


# Rotate the object a bit

rot = la.quat_from_euler((0.71, 0.1), order="XY")
for obj in scene.children:
    obj.local.rotation = la.quat_mul(rot, obj.local.rotation)


canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    run()
