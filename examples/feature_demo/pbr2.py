"""
PBR Rendering 2
===============

This example shows the lighting rendering affect of materials with different
metalness and roughness. Every second sphere has an IBL environment map on it.
"""

# sphinx_gallery_pygfx_docs = 'animate 4s'
# sphinx_gallery_pygfx_test = 'run'

import math
from time import perf_counter

import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run

import pygfx as gfx
import pylinalg as la

# Init
canvas = WgpuCanvas(size=(640, 480), title="gfx_pbr")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# Lights
scene.add(gfx.AmbientLight("#fff", 0.2))
directional_light = gfx.DirectionalLight("#fff", 3)
directional_light.local.position = la.vec_normalize((1, 1, 1))
scene.add(directional_light)
point_light = gfx.PointLight("#fff", 3)
scene.add(point_light)
point_light.add(gfx.PointLightHelper(4))

# Read cube image and turn it into a 3D image (a 4d array)
env_img = iio.imread("imageio:meadow_cube.jpg")
cube_size = env_img.shape[1]
env_img.shape = 6, cube_size, cube_size, env_img.shape[-1]

# Create environment map
env_tex = gfx.Texture(
    env_img, dim=2, size=(cube_size, cube_size, 6), generate_mipmaps=True
)

# Apply env map to skybox
background = gfx.Background(None, gfx.BackgroundSkyboxMaterial(map=env_tex))
scene.add(background)

# Now create spheres ...

cube_width = 400
numbers_per_side = 5
sphere_radius = (cube_width / numbers_per_side) * 0.8 * 0.5
step_size = 1.0 / numbers_per_side

geometry = gfx.sphere_geometry(sphere_radius, 32, 16)

index = 0
alpha = 0.0
while alpha <= 1.0:
    beta = 0.0
    while beta <= 1.0:
        gamma = 0.0
        while gamma <= 1.0:
            material = gfx.MeshStandardMaterial(
                color=gfx.Color.from_physical(
                    *gfx.Color.from_hsl(alpha, 0.5, gamma * 0.5 + 0.1)
                ),
                metalness=beta,
                roughness=1.0 - alpha,
            )

            if index % 2 != 0:
                material.env_map = env_tex

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

# Create camera and controls
camera = gfx.PerspectiveCamera(45, 640 / 480)
camera.show_object(scene, view_dir=(-2, -1.5, -3), scale=1.2)

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
