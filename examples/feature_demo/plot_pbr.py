"""
PBR Rendering 1
===============


This example shows a complete PBR rendering effect.
The cubemap of skybox is also the environment cubemap of the helmet.
"""

from pathlib import Path

import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


# Init
canvas = WgpuCanvas(size=(640, 480), title="gfx_pbr")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# Create camera and controller
camera = gfx.PerspectiveCamera(45, 640 / 480, 0.25, 20)
camera.position.set(-1.8, 0.6, 2.7)
controller = gfx.OrbitController(camera.position.clone())
controller.add_default_event_handlers(renderer, camera)

# Read cube image and turn it into a 3D image (a 4d array)
env_img = iio.imread("imageio:meadow_cube.jpg")
cube_size = env_img.shape[1]
env_img.shape = 6, cube_size, cube_size, env_img.shape[-1]

# Create environment map
env_tex = gfx.Texture(env_img, dim=2, size=(cube_size, cube_size, 6))
env_tex.generate_mipmaps = True
env_view = env_tex.get_view(
    view_dim="cube", layer_range=range(6), address_mode="repeat", filter="linear"
)

# Apply env map to skybox
background = gfx.Background(None, gfx.BackgroundSkyboxMaterial(map=env_view))
scene.add(background)

# Load meshes, and apply env map
# Note that this lits the helmet already
gltf_path = (
    Path(__file__).parents[1]
    / "models"
    / "DamagedHelmet"
    / "glTF"
    / "DamagedHelmet.gltf"
)
meshes = gfx.load_scene(gltf_path)
scene.add(*meshes)
m = meshes[0]  # this example has just one mesh
m.material.env_map = env_view

# Add extra light more or less where the sun seems to be in the skybox
scene.add(gfx.SpotLight(color="#444", position=(-500, 1000, -1000)))


def animate():
    controller.update_camera(camera)
    renderer.render(scene, camera)


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
