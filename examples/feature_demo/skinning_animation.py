"""
Skinning Animation
================

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

# sphinx_gallery_pygfx_docs = 'animate 4s'
# sphinx_gallery_pygfx_test = 'run'

import math, time
import numpy as np
import pygfx as gfx
from wgpu.gui.auto import WgpuCanvas, run
from  pygfx.utils.gltf_loader import GLTF, print_tree


gltf_path = model_dir / "Michelle.glb"

# meshes = load_mesh(gltf_path)


scene = gfx.Scene()

# scene.add(*meshes)

gltf = GLTF.load(gltf_path)

# print_tree(gltf.scene)

mesh_obj = gltf.scene.children[0]

canvas = WgpuCanvas(size=(640, 480), max_fps=60, title="Skinnedmesh")

renderer = gfx.WgpuRenderer(canvas)
camera = gfx.PerspectiveCamera(75, 640 / 480, depth_range=(0.1, 1000))
camera.local.position = (0, 100, 200)
camera.look_at((0, 100, 0))
scene = gfx.Scene()

scene.add(gfx.AmbientLight(), gfx.DirectionalLight())

mesh_obj.local.scale = ( 1, 1, 1 )


skeleton_helper = gfx.SkeletonHelper(gltf.scene)
scene.add(skeleton_helper)

scene.add(gltf.scene)

# xyz = gfx.AxesHelper( 20 )
# scene.add( xyz )


gfx.OrbitController(camera, register_events=renderer)

def animate():
    t = time.time()
    mesh_obj.children[0].skeleton.update()
    skeleton_helper.update()

    renderer.render(scene, camera)
    canvas.request_draw()

if __name__ == "__main__":
    renderer.request_draw(animate)
    run()