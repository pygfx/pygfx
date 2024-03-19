"""
Instancing
==========

Example rendering the same mesh object multiple times, using instancing.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

im = iio.imread("imageio:bricks.jpg").astype(np.float32) / 255
tex = gfx.Texture(im, dim=2)

geometry = gfx.torus_knot_geometry(1, 0.3, 128, 32)
geometry.texcoords.data[:, 0] *= 10  # stretch the texture
material = gfx.MeshPhongMaterial(map=tex)
obj = gfx.InstancedMesh(geometry, material, 100)
scene.add(obj)


# Set matrices. Note that these are sub-transforms of the mesh's own matrix.
for y in range(10):
    for x in range(10):
        m = la.mat_from_translation((y * 2, x * 2, 0))
        obj.set_matrix_at(x + y * 10, m)


camera = gfx.PerspectiveCamera(70, 1)
camera.local.position = (9, 9, 15)

scene.add(gfx.AmbientLight())
scene.add(camera.add(gfx.DirectionalLight()))


def animate():
    rot = la.quat_from_euler((0.0071, 0.01), order="XY")
    obj.local.rotation = la.quat_mul(rot, obj.local.rotation)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
