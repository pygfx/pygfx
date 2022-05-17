"""
Example demonstrating rendering the same scene into two different canvases.
"""

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


# Create two canvases and two renderers

canvas_a = WgpuCanvas(size=(500, 300))
canvas_b = WgpuCanvas(size=(300, 500))

renderer_b = gfx.renderers.WgpuRenderer(canvas_b)
renderer_a = gfx.renderers.WgpuRenderer(canvas_a)

# Compose a 3D scene with 2 objects

scene = gfx.Scene()

geometry1 = gfx.box_geometry(200, 200, 200)
material1 = gfx.MeshPhongMaterial(color=(1, 1, 0, 1.0))
cube = gfx.Mesh(geometry1, material1)
scene.add(cube)

positions = np.array(
    [[-1, -1, 0], [-1, +1, 0], [+1, +1, 0], [+1, -1, 0], [-1, -1, 0], [+1, +1, 0]],
    np.float32,
)
geometry2 = gfx.Geometry(positions=positions * 250)
material2 = gfx.LineMaterial(thickness=5.0, color=(0.8, 0.0, 0.2, 1.0))
line = gfx.Line(geometry2, material2)
scene.add(line)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 400

# add a directional light to illuminate the scene
light = gfx.DirectionalLight(color=(1, 1, 1, 1))
camera.add(light)

# Define animation functions. Each renders the scene into its own canvas.


def animate_a():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    cube.rotation.multiply(rot)
    renderer_a.render(scene, camera)
    canvas_a.request_draw()


def animate_b():
    renderer_b.render(scene, camera)
    canvas_b.request_draw()


if __name__ == "__main__":
    canvas_a.request_draw(animate_a)
    canvas_b.request_draw(animate_b)
    run()
