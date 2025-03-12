"""
PBR Instanced Rendering
=======================

This example shows how to use PBR rendering with instanced meshes.
"""

################################################################################
# .. note::
#
#   To run this example, you need a model from the source repo's example
#   folder. If you are running this example from a local copy of the code (dev
#   install) no further actions are needed. Otherwise, you may have to replace
#   the path below to point to the location of the model.

import os
import pylinalg as la
import numpy as np
from pathlib import Path

try:
    # modify this line if your model is located elsewhere
    model_dir = Path(__file__).parents[1] / "data"
except NameError:
    # compatibility with sphinx-gallery
    model_dir = Path(os.getcwd()).parent / "data"


################################################################################
# Once the path is set correctly, you can use the model as follows:

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx

# Init
canvas = WgpuCanvas(size=(640, 480), title="gfx_pbr")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

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

# Load meshes, and apply env map
# Note that this lights the helmet already
gltf_path = model_dir / "DamagedHelmet" / "glTF" / "DamagedHelmet.gltf"

gltf = gfx.load_gltf(gltf_path)
# gfx.print_scene_graph(gltf.scene) # Uncomment to see the tree structure

m = gltf.scene.children[0]

m.material.env_map = env_tex

instanced_mesh = gfx.InstancedMesh(m.geometry, m.material, 100)

instanced_mesh.local.matrix = m.local.matrix

# Set matrices. Note that these are sub-transforms of the mesh's own matrix.
for y in range(10):
    for x in range(10):
        m_t = la.mat_from_translation((y * 2, x * 2, 0))
        m_r = la.mat_from_euler((np.pi / 5 * x, np.pi / 5 * y, 0))
        m = m_t @ m_r
        instanced_mesh.set_matrix_at(x + y * 10, m)


scene.add(instanced_mesh)

# Add extra light more or less where the sun seems to be in the skybox
light = gfx.SpotLight(color="#444")
light.local.position = (-500, 1000, -1000)
scene.add(light)


# Create camera and controller
camera = gfx.PerspectiveCamera(45, 640 / 480)
camera.local.position = (15, 15, 15)
camera.show_pos((10, 0, 10))
controller = gfx.OrbitController(camera, register_events=renderer)


if __name__ == "__main__":
    renderer.request_draw(lambda: renderer.render(scene, camera))
    run()
