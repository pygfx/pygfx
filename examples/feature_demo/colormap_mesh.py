"""
Colormap Mesh
=============

Example demonstrating different colormap dimensions on a mesh, and
per-vertex colors as a bonus.

The default visiualization is a mesh, but by (un)commenting a few lines,
this can also be applied for points and lines.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
import imageio.v3 as iio
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
    return gfx.MeshPhongMaterial(**kwargs)
    # return gfx.PointsMaterial(size=10, **kwargs)
    # return gfx.LineArrowMaterial(thickness=5, **kwargs)


geometry = get_geometry()

camera = gfx.OrthographicCamera(16, 3)


# === 1D colormap
#
# For the 1D texcoords we use the second dimension of the default
# texcoords, which runs from the top to the bottom of the cylinder. The
# 1D colormap runs from yellow to cyan.

texcoords1 = geometry.texcoords.data[:, 1].copy()

cmap1 = np.array([(1, 1, 0), (0, 1, 1)], np.float32)
tex1 = gfx.Texture(cmap1, dim=1)

ob1 = WobjectClass(
    get_geometry(texcoords=gfx.Buffer(texcoords1)),
    MaterialClass(map=tex1),
)
scene.add(ob1)
ob1.local.x = -6


# === 2D colormap
#
# For the 2D texcoords we use the default texcoords. For the 2D colormap
# we use an image texture.

texcoords2 = geometry.texcoords.data.copy()

cmap2 = iio.imread("imageio:chelsea.png").astype(np.float32) / 255
tex2 = gfx.Texture(cmap2, dim=2)

ob2 = WobjectClass(
    get_geometry(texcoords=gfx.Buffer(texcoords2)),
    MaterialClass(map=tex2),
)
scene.add(ob2)
ob2.local.x = -2


# === 3D colormap
#
# For the 3D texcoords we use (a scaled version of) the positions. For
# the colormap we use a volume (a 3D image). In effect, the edge of the
# mesh gets a color that corresponds to the value of the volume at that
# position. This can be seen as a specific (maybe somewhat odd) type
# of volume rendering.

texcoords3 = geometry.positions.data * 0.4 + 0.5

cmap3 = iio.imread("imageio:stent.npz").astype(np.float32) / 1000
tex3 = gfx.Texture(cmap3, dim=3)

ob3 = WobjectClass(
    get_geometry(texcoords=gfx.Buffer(texcoords3)),
    MaterialClass(map=tex3),
)
scene.add(ob3)
ob3.local.x = +2


# === Per vertex coloring
#
# To specify a color for each vertex, provide a geometry.colors buffer and
# set the material.color_mode to 'vertex'. We use the normals as input.

colors = geometry.normals.data * 0.4 + 0.5
colors = colors[:, :3]  # Colors can be Nx1, Nx2, Nx3, Nx4

ob4 = WobjectClass(
    get_geometry(colors=gfx.Buffer(colors)),
    MaterialClass(color_mode="vertex"),
)
scene.add(ob4)
ob4.local.x = +6

scene.add(gfx.AmbientLight())
light = gfx.DirectionalLight()
light.local.z = 1
scene.add(light)


def animate():
    rot = la.quat_from_euler((0.0071, 0.01), order="XY")
    for obj in scene.children:
        obj.local.rotation = la.quat_mul(rot, obj.local.rotation)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
