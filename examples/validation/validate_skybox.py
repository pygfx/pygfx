"""
Validate Skybox
===============

This validates that the skybox background is rendered correctly,
as well as that the environment-map matches that background. Therefore
it also covers the internal cube-camera-renderer.

Note that the blue -Z side is marked "back", but since camera's look down the
-Z direction, it'd be what the camera would see if it was not rotated.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import os

import numpy as np
import imageio.v3 as iio
import pygfx as gfx
from pathlib import Path
from wgpu.gui.auto import WgpuCanvas, run

try:
    data_dir = Path(__file__).parents[1] / "data"
except NameError:
    data_dir = Path(os.getcwd()).parent / "data"  # compat with sphinx-gallery

data = iio.imread(data_dir / "cubemap.jpg")

h = data.shape[0] // 3
w = data.shape[1] // 4

"""
The layout of the example cubemap looks like this:
 ┌────┬────┬────┬────┐
 │    │ +Y │    │    │
 ├────┼────┼────┼────┤
 │ -X │ +Z │ +X │ -Z │
 ├────┼────┼────┼────┤
 │    │ -Y │    │    │
 └────┴────┴────┴────┘
 """
posx = np.ascontiguousarray(data[1 * h : 2 * h, 2 * w : 3 * w])
negx = np.ascontiguousarray(data[1 * h : 2 * h, 0 * w : 1 * w])
posy = np.ascontiguousarray(data[0 * h : 1 * h, 1 * w : 2 * w])
negy = np.ascontiguousarray(data[2 * h : 3 * h, 1 * w : 2 * w])
posz = np.ascontiguousarray(data[1 * h : 2 * h, 1 * w : 2 * w])
negz = np.ascontiguousarray(data[1 * h : 2 * h, 3 * w : 4 * w])

datas = [posx, negx, posy, negy, posz, negz]

tex = gfx.Texture(np.stack(datas, axis=0), dim=2, size=(w, h, 6), generate_mipmaps=True)

canvas = WgpuCanvas(size=(640, 640))
renderer = gfx.renderers.WgpuRenderer(canvas)

scene = gfx.Scene()

background = gfx.Background(None, gfx.BackgroundSkyboxMaterial(map=tex))
scene.add(background)

mesh = gfx.Mesh(
    gfx.sphere_geometry(2, 64, 64),
    gfx.MeshStandardMaterial(roughness=0.01, metalness=1, side="Front"),
)
mesh.material.env_map = tex
scene.add(mesh)

camera = gfx.PerspectiveCamera(90)
camera.local.position = (2, 3, 4)
camera.show_pos(mesh)

controller = gfx.OrbitController(camera, register_events=renderer)

renderer.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    run()
