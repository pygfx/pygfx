"""
This example shows a dynamic environment map used for MeshStandardMaterial.
The environment map is automatically be updated from the scene by a CubeCamera.
"""

import time
import math

import imageio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
from pygfx.utils.cube_camera import CubeCamera

renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
scene = gfx.Scene()

material = gfx.MeshStandardMaterial(roughness=0.05, metalness=1)
material.side = "Front"

sphere = gfx.Mesh(
    gfx.sphere_geometry(15, 64, 64),
    material,
)

scene.add(sphere)

material2 = gfx.MeshStandardMaterial(roughness=0.1, metalness=0)
material2.side = "Front"

cube = gfx.Mesh(gfx.box_geometry(15, 15, 15), material2)
scene.add(cube)

torus = gfx.Mesh(gfx.torus_knot_geometry(8, 3, 128, 16), material2)
scene.add(torus)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 75

controller = gfx.OrbitController(camera.position.clone())
controller.add_default_event_handlers(renderer, camera)

# Read cube image and turn it into a 3D image (a 4d array)
env_img = imageio.imread("imageio:meadow_cube.jpg")
cube_size = env_img.shape[1]
env_img.shape = 6, cube_size, cube_size, env_img.shape[-1]

# Create environment map
env_tex = gfx.Texture(
    env_img, dim=2, size=(cube_size, cube_size, 6), generate_mipmaps=True
)

env_view = env_tex.get_view(
    view_dim="cube", layer_range=range(6), address_mode="repeat", filter="linear"
)

background = gfx.Background(None, gfx.BackgroundSkyboxMaterial(map=env_view))
scene.add(background)

material2.env_map = env_view

cube_camera = CubeCamera()

gen_env = gfx.Texture(
    dim=2, size=(512, 512, 6), format="rgba8unorm-srgb", generate_mipmaps=True
)

gen_env_view = gen_env.get_view(
    view_dim="cube", layer_range=range(6), address_mode="repeat", filter="linear"
)

material.env_map = gen_env_view


def animate():
    controller.update_camera(camera)
    t = time.time()

    cube.position.x = math.cos(t) * 30
    cube.position.y = math.sin(t) * 30
    cube.position.z = math.sin(t) * 30

    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.02, 0.03))
    cube.rotation.multiply(rot)

    torus.position.x = math.cos(t + 10) * 30
    torus.position.y = math.sin(t + 10) * 30
    torus.position.z = math.sin(t + 10) * 30

    torus.rotation.multiply(rot)

    cube_camera.update(renderer, scene, target=gen_env)

    renderer.render(scene, camera)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
