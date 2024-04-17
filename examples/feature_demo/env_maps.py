"""
Environment Map Effects
=======================

This example demonstrates the environment mapping effects
of the "CUBE-REFLECTION" and "CUBE-REFRACTION" mapping modes
in "MeshBasicMaterial", "MeshPhongMaterial", and "MeshStandardMaterial"
for a given object.

"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import os
import math

import imageio
import trimesh
from pathlib import Path
import pylinalg as la
from wgpu.gui.auto import WgpuCanvas, run

import pygfx as gfx

try:
    # modify this line if your model is located elsewhere
    model_dir = Path(__file__).parents[1] / "data"
except NameError:
    # compatibility with sphinx-gallery
    model_dir = Path(os.getcwd()).parent / "data"

TEAPOT = model_dir / "teapot.stl"

renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
scene = gfx.Scene()

dir_light = gfx.DirectionalLight(1, 2)
dir_light.local.position = (0, 0, 1)

scene.add(gfx.AmbientLight(1, 0.2), dir_light)

# Read cube image and turn it into a 3D image (a 4d array)
env_img = imageio.imread("imageio:meadow_cube.jpg")
cube_size = env_img.shape[1]
env_img.shape = 6, cube_size, cube_size, env_img.shape[-1]

# Create environment map
env_tex = gfx.Texture(
    env_img, dim=2, size=(cube_size, cube_size, 6), generate_mipmaps=True
)

teapot = trimesh.load(TEAPOT)

rot = la.quat_from_euler(-math.pi / 2, order="X")


def add_mesh(material, env_mapping_mode, pos_x, pos_y):
    material.env_map = env_tex
    material.env_mapping_mode = env_mapping_mode

    mesh = gfx.Mesh(gfx.geometry_from_trimesh(teapot), material)

    mesh.local.position = (pos_x, pos_y, 0)
    mesh.local.rotation = la.quat_mul(rot, mesh.local.rotation)
    scene.add(mesh)


# BasicMaterial - CUBE-REFRACTION
material = gfx.MeshBasicMaterial()
add_mesh(material, "CUBE-REFRACTION", -60, 0)

# BasicMaterial - CUBE-REFLECTION
material = gfx.MeshBasicMaterial()
add_mesh(material, "CUBE-REFLECTION", -60, -60)

# PhongMaterial - CUBE-REFRACTION
material = gfx.MeshPhongMaterial()
material.env_combine_mode = "MIX"  # can be "MIX", "MULTIPLY" or "ADD"
add_mesh(material, "CUBE-REFRACTION", 0, 0)

# PhongMaterial - CUBE-REFLECTION
material = gfx.MeshPhongMaterial()
material.env_combine_mode = "MIX"  # can be "MIX", "MULTIPLY" or "ADD"
add_mesh(material, "CUBE-REFLECTION", 0, -60)

# StandardMaterial - CUBE-REFRACTION
material = gfx.MeshStandardMaterial(roughness=0.1, metalness=1.0)
add_mesh(material, "CUBE-REFRACTION", 60, 0)

# StandardMaterial - CUBE-REFLECTION
material = gfx.MeshStandardMaterial(roughness=0.1, metalness=1.0)
add_mesh(material, "CUBE-REFLECTION", 60, -60)

background = gfx.Background(None, gfx.BackgroundSkyboxMaterial(map=env_tex))
scene.add(background)

camera = gfx.PerspectiveCamera(45, 16 / 9)
camera.show_object(scene, view_dir=(0, 0, -300))
controller = gfx.OrbitController(camera, register_events=renderer)

if __name__ == "__main__":
    renderer.request_draw(lambda: renderer.render(scene, camera))
    run()
