"""
Toon Rendering 2
================

This example shows the toon rendering affect of materials with different
colors and gradient maps.
"""

# sphinx_gallery_pygfx_docs = 'animate 4s'
# sphinx_gallery_pygfx_test = 'run'

import math
from time import perf_counter
import numpy as np

from wgpu.gui.auto import WgpuCanvas, run

import pygfx as gfx

# Init
canvas = WgpuCanvas(size=(640, 480), title="gfx_toon")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# Lights
scene.add(gfx.AmbientLight("#c1c1c1", 3))

point_light = gfx.PointLight("#fff", 2, distance=800, decay=0)
scene.add(point_light)
point_light.add(gfx.PointLightHelper(4))

background = gfx.Background.from_color("#444488")
scene.add(background)

# Now create spheres ...

cube_width = 400
numbers_per_side = 5
sphere_radius = (cube_width / numbers_per_side) * 0.8 * 0.5
step_size = 1.0 / numbers_per_side

geometry = gfx.sphere_geometry(sphere_radius, 32, 16)

index = 0
alpha = 0.0
alpha_index = 0
while alpha <= 1.0:
    colors = np.arange(alpha_index + 2) / (alpha_index + 1) * 255
    gradient_map_data = np.array(colors, dtype=np.uint8).reshape(1, -1, 1)  # H,W,C
    gradient_map = gfx.Texture(gradient_map_data, dim=2)

    beta = 0.0
    while beta <= 1.0:
        gamma = 0.0
        while gamma <= 1.0:
            diffuse_color = gfx.Color.from_physical(
                *gfx.Color.from_hsl(alpha, 0.5, gamma * 0.5 + 0.1)
            )
            diffuse_color = diffuse_color * (1.0 - beta * 0.2)
            material = gfx.MeshToonMaterial(
                color=diffuse_color,
                gradient_map=gradient_map,
            )

            mesh = gfx.Mesh(geometry, material)

            mesh.local.position = (
                alpha * 400 - 200,
                beta * 400 - 200,
                gamma * 400 - 200,
            )
            scene.add(mesh)
            index += 1

            gamma += step_size
        beta += step_size
    alpha += step_size
    alpha_index += 1


def add_label(text, pos):
    text = gfx.Text(
        text=text,
        font_size=20,
        text_align="center",
        anchor="middle-center",
        material=gfx.TextMaterial(outline_color="#000", outline_thickness=0.5),
    )
    text.local.position = pos
    scene.add(text)


add_label("-gradientMap", (-350, 0, 0))
add_label("+gradientMap", (350, 0, 0))

add_label("+diffuse", (0, 0, 300))
add_label("-diffuse", (0, 0, -300))

# Create camera and controls
camera = gfx.PerspectiveCamera(45, 640 / 480)
camera.show_object(scene, view_dir=(-2, -1.5, -3), scale=1)

controller = gfx.OrbitController(camera, register_events=renderer)

t0 = perf_counter()


def animate():
    t = perf_counter() - t0

    t = t * 0.25

    point_light.local.position = (
        math.sin(t * 7) * 300,
        math.cos(t * 5) * 400,
        math.cos(t * 3) * 300,
    )

    renderer.render(scene, camera)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
