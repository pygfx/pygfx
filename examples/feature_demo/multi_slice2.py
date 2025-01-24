"""
Mesh and Volume Slicing 2
=========================


Slice a volume and a mesh through the three primary planes (XY, XZ, YZ).
This example uses Volume object with a VolumeSliceMaterial, which
produces an implicit geometry defined by the volume data.
See multi_slice1.py for a more generic approach.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from time import time

import imageio.v3 as iio
import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
from skimage.measure import marching_cubes
import pylinalg as la

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

dark_gray = np.array((169, 167, 168, 255)) / 255
light_gray = np.array((100, 100, 100, 255)) / 255
background = gfx.Background.from_color(light_gray, dark_gray)
scene.add(background)

scene.add(gfx.AxesHelper(size=50))

vol = iio.imread("imageio:stent.npz").astype("float32")
tex = gfx.Texture(vol, dim=3)

surface = marching_cubes(vol[0:], 200)
geo = gfx.Geometry(
    positions=np.fliplr(surface[0]), indices=surface[1], normals=surface[2]
)
mesh = gfx.Mesh(
    geo, gfx.MeshSliceMaterial(plane=(0, 0, -1, vol.shape[0] / 2), color=(1, 1, 0, 1))
)
scene.add(mesh)

planes = []
for dim in [0, 1, 2]:  # xyz
    abcd = [0, 0, 0, 0]
    abcd[dim] = -1
    abcd[-1] = vol.shape[2 - dim] / 2
    material = gfx.VolumeSliceMaterial(clim=(0, 2000), plane=abcd)
    plane = gfx.Volume(gfx.Geometry(grid=tex), material)
    planes.append(plane)
    scene.add(plane)


camera = gfx.PerspectiveCamera(50)
camera.show_object(scene, (-1, -1, -1), up=(0, 0, 1))

controller = gfx.OrbitController(camera, register_events=renderer)

# Add a slight tilt. This is to show that the slices are still orthogonal
# to the world coordinates.
for ob in [*planes, mesh]:
    ob.local.rotation = la.quat_from_axis_angle((1, 0, 0), 0.1)


def animate():
    t = np.cos(time() / 2) * 0.5 + 0.5  # 0..1
    planes[2].material.plane = 0, 0, -1, t * vol.shape[0]
    mesh.material.plane = 0, 0, -1, (1 - t) * vol.shape[0]

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
