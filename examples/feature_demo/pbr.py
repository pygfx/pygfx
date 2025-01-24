"""
PBR Rendering 1
===============

This example shows a complete PBR rendering effect.
The cubemap of skybox is also the environment cubemap of the helmet.
"""

################################################################################
# .. note::
#
#   To run this example, you need a model from the source repo's example
#   folder. If you are running this example from a local copy of the code (dev
#   install) no further actions are needed. Otherwise, you may have to replace
#   the path below to point to the location of the model.

import os
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

scene.add(gltf.scene)

# Add extra light more or less where the sun seems to be in the skybox
light = gfx.SpotLight(color="#444")
light.local.position = (-500, 1000, -1000)
scene.add(light)

# Create camera and controller
camera = gfx.PerspectiveCamera(45, 640 / 480)
camera.show_object(m, view_dir=(1.8, -0.6, -2.7))
controller = gfx.OrbitController(camera, register_events=renderer)


if __name__ == "__main__":
    renderer.request_draw(lambda: renderer.render(scene, camera))
    run()
