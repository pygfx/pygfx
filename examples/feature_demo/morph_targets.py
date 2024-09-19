"""
Morph Targets
=============

This example demonstrates how to use morph targets to animate a mesh.
"""

# sphinx_gallery_pygfx_docs = 'animate 4s'
# sphinx_gallery_pygfx_test = 'run'

import math
import time
import numpy as np
import pygfx as gfx
import pylinalg as la
from wgpu.gui.auto import WgpuCanvas, run


def create_geometry():
    geometry = gfx.box_geometry(2, 2, 2, 32, 32, 32)

    # the original positions of the cube's vertices
    positions = geometry.positions

    # for the first morph target we'll move the cube's vertices onto the surface of a sphere
    sphere_positions = np.zeros_like(positions.data)

    # for the second morph target, we'll twist the cubes vertices
    twist_positions = np.zeros_like(positions.data)

    for i in range(positions.nitems):
        x, y, z = positions.data[i]

        # sphere
        sphere_positions[i] = (
            x * math.sqrt(1 - (y**2 / 2) - (z**2 / 2) + (y**2 * z**2 / 3)),
            y * math.sqrt(1 - (z**2 / 2) - (x**2 / 2) + (z**2 * x**2 / 3)),
            z * math.sqrt(1 - (x**2 / 2) - (y**2 / 2) + (x**2 * y**2 / 3)),
        )

        # twist
        # stretch along the x-axis so we can see the twist better
        twist_positions[i] = la.vec_transform_quat(
            (2 * x, y, z), la.quat_from_axis_angle((1, 0, 0), math.pi * x / 2)
        )

    geometry.morph_positions = []

    # add the spherical positions as the first morph target
    geometry.morph_positions.append(sphere_positions)

    # add the twist positions as the second morph target
    geometry.morph_positions.append(twist_positions)

    return geometry


canvas = WgpuCanvas(size=(640, 480), max_fps=60, title="Morph Targets")

renderer = gfx.WgpuRenderer(canvas)

camera = gfx.PerspectiveCamera(45, 640 / 480, depth_range=(0.1, 100))

scene = gfx.Scene()

geometry = create_geometry()
material = gfx.MeshNormalMaterial()

mesh = gfx.Mesh(geometry, material)

scene.add(mesh)

camera.show_object(mesh, scale=2.5)
controller = gfx.OrbitController(camera, register_events=renderer)

gfx.OrbitController(camera, register_events=renderer)


def animate():
    t = time.time()

    mesh.morph_target_influences = [
        0.5 + 0.5 * math.sin(t),
        0.5 + 0.5 * math.cos(t + 1),
    ]

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
