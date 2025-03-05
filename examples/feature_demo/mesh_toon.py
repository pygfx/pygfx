"""
Toon Rendering 1
================

This example demonstrates the affect of the `MeshToonMaterial`
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

geometry = gfx.torus_knot_geometry(1, 0.3, 128, 64)

material = gfx.MeshToonMaterial(color="#049ef4")

gradient_map = np.array([64, 128, 255], dtype=np.uint8).reshape(
    1, 3, 1
)  # H,W,C = 1,3,1

material.gradient_map = gfx.Texture(gradient_map, dim=2)

obj = gfx.Mesh(geometry, material)
scene.add(obj)

camera = gfx.PerspectiveCamera(70, 1)
camera.local.z = 4

dl = gfx.DirectionalLight(intensity=3.0)
dl.local.position = (0, 1, 0)

scene.add(dl)


def animate():
    rot = la.quat_from_euler((0.0071, 0.01), order="XY")
    obj.local.rotation = la.quat_mul(rot, obj.local.rotation)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
