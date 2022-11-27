"""

PBR Rendering 2
===============

This example shows the lighting rendering affect of materials with different
metalness and roughness. Every second sphere has an IBL environment map on it.
"""
# run_example = false
# sphinx_gallery_pygfx_render = True

import time
import math
from colorsys import hls_to_rgb

import imageio.v3 as iio
import pygfx as gfx
from wgpu.gui.auto import WgpuCanvas, run


# Init
canvas = WgpuCanvas(size=(640, 480), title="gfx_pbr")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# Create camera and controls
camera = gfx.PerspectiveCamera(45, 640 / 480, 1, 2500)
camera.position.set(0, 400, 400 * 3)
controller = gfx.OrbitController(camera.position.clone())
controller.add_default_event_handlers(renderer, camera)

# Lights
scene.add(gfx.AmbientLight("#fff", 0.2))
directional_light = gfx.DirectionalLight("#fff", 3)
directional_light.position.set(1, 1, 1).normalize()
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
env_view = env_tex.get_view(
    view_dim="cube", layer_range=range(6), address_mode="repeat", filter="linear"
)

# Apply env map to skybox
background = gfx.Background(None, gfx.BackgroundSkyboxMaterial(map=env_view))
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
                color=hls_to_rgb(alpha, 0.5, gamma * 0.5 + 0.1),
                metalness=beta,
                roughness=1.0 - alpha,
            )

            if index % 2 != 0:
                material.env_map = env_view

            mesh = gfx.Mesh(geometry, material)

            mesh.position.x = alpha * 400 - 200
            mesh.position.y = beta * 400 - 200
            mesh.position.z = gamma * 400 - 200
            scene.add(mesh)
            index += 1

            gamma += step_size
        beta += step_size
    alpha += step_size


def animate():
    timer = time.time() * 0.25
    controller.update_camera(camera)

    point_light.position.x = math.sin(timer * 7) * 300
    point_light.position.y = math.cos(timer * 5) * 400
    point_light.position.z = math.cos(timer * 3) * 300

    renderer.render(scene, camera)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
