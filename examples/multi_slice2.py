"""
Slice a volume and a mesh through the three primary planes (XY, XZ, YZ).
This example uses Volume object with a VolumeSliceMaterial, which
produces an implicit geometry defined by the volume data.
See multi_slice1.py for a more generic approach.
"""

from time import time

import imageio
import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
from skimage.measure import marching_cubes


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

background = gfx.Background(None, gfx.BackgroundMaterial((0, 1, 0, 1), (0, 1, 1, 1)))
scene.add(background)

scene.add(gfx.AxesHelper(length=50))

vol = imageio.volread("imageio:stent.npz").astype("float32")
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


# camera = gfx.PerspectiveCamera(70, 16 / 9)
camera = gfx.OrthographicCamera(200, 200)
camera.position.set(170, 170, 170)
controls = gfx.OrbitControls(
    camera.position.clone(),
    gfx.linalg.Vector3(64, 64, 128),
    up=gfx.linalg.Vector3(0, 0, 1),
    zoom_changes_distance=False,
)
controls.add_default_event_handlers(canvas, camera)

# Add a slight tilt. This is to show that the slices are still orthogonal
# to the world coordinates.
for ob in planes + [mesh]:
    ob.rotation.set_from_axis_angle(gfx.linalg.Vector3(1, 0, 0), 0.1)


def animate():
    t = np.cos(time() / 2) * 0.5 + 0.5  # 0..1
    planes[2].material.plane = 0, 0, -1, t * vol.shape[0]
    mesh.material.plane = 0, 0, -1, (1 - t) * vol.shape[0]

    controls.update_camera(camera)
    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
