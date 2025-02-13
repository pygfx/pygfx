"""
Iridescence
===========

This example demonstrates iridescence on a set of spheres.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from time import perf_counter
import numpy as np

import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx

# Init
canvas = WgpuCanvas(size=(640, 480), title="Iridescence2")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# Lights
scene.add(gfx.AmbientLight("#c1c1c1", 0.1))

background = gfx.Background.from_color("#444488")
scene.add(background)

# Read cube image and turn it into a 3D image (a 4d array)
env_img = iio.imread("imageio:meadow_cube.jpg")
cube_size = env_img.shape[1]
env_img.shape = 6, cube_size, cube_size, env_img.shape[-1]

# Create environment map, use for IBL.
env_tex = gfx.Texture(
    env_img, dim=2, size=(cube_size, cube_size, 6), generate_mipmaps=True
)

# Now create spheres ...

cube_width = 400
numbers_per_side = 5
sphere_radius = (cube_width / numbers_per_side) * 0.8 * 0.4
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
            material = gfx.MeshPhysicalMaterial(
                iridescence_ior=1.0 + gamma,  # from 1.0 to 2.0
                iridescence_thickness_range=(
                    100,
                    beta * 600 + 100,
                ),  # film thickness from 100 to 700
                ior=1.0 + alpha,  # from 1.0 to 2.0
                env_map=env_tex,
                iridescence=1.0,
                roughness=0.2,
                metalness=0.3,
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


# Create camera and controls
camera = gfx.PerspectiveCamera(45, 640 / 480)
camera.show_object(scene, view_dir=(-2, -1.5, -3), scale=1.2)

controller = gfx.OrbitController(camera, register_events=renderer)

t0 = perf_counter()


def animate():
    renderer.render(scene, camera)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
