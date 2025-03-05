"""
Detached Bind Mode
==================

This example demonstrates how to sharing a skeleton across multiple skinned meshes with detached bind mode.

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

import numpy as np
import pygfx as gfx
from wgpu.gui.auto import WgpuCanvas, run

gltf_path = model_dir / "Michelle.glb"

canvas = WgpuCanvas(size=(640, 480), max_fps=-1, title="Skinnedmesh", vsync=False)

renderer = gfx.WgpuRenderer(canvas)
camera = gfx.PerspectiveCamera(75, 640 / 480, depth_range=(0.1, 1000))
camera.local.position = (0, 100, 200)
camera.look_at((0, 100, 0))
scene = gfx.Scene()

scene.add(gfx.AmbientLight(), gfx.DirectionalLight())


gltf = gfx.load_gltf(gltf_path, quiet=True)

# gfx.print_scene_graph(gltf.scene) # Uncomment to see the tree structure

# Group[Scene]
# - WorldObject[Character]
# - - SkinnedMesh[Ch03]
# - - Bone[mixamorig:Hips]
# - - - ...

model_obj = gltf.scene.children[0]
model_obj.local.scale = (1, 1, 1)
scene.add(model_obj)

skinnedmesh = model_obj.children[0]
skeleton = skinnedmesh.skeleton

skinnedmesh1 = gfx.SkinnedMesh(skinnedmesh.geometry, skinnedmesh.material)
skinnedmesh1.local.x = -80
skinnedmesh1.bind_mode = gfx.BindMode.detached
skinnedmesh1.bind(skeleton, np.eye(4))

scene.add(skinnedmesh1)

skinnedmesh2 = gfx.SkinnedMesh(skinnedmesh.geometry, skinnedmesh.material)
skinnedmesh2.local.x = 80
skinnedmesh2.bind_mode = gfx.BindMode.detached
skinnedmesh2.bind(skeleton, np.eye(4))

scene.add(skinnedmesh2)

skeleton_helper = gfx.SkeletonHelper(model_obj)
scene.add(skeleton_helper)

gfx.OrbitController(camera, register_events=renderer)

stats = gfx.Stats(viewport=renderer)

clock = gfx.Clock()
mixer = gfx.AnimationMixer()

action_clip = gltf.animations[0]
action = mixer.clip_action(action_clip)
action.play()


def animate():
    dt = clock.get_delta()
    mixer.update(dt)

    with stats:
        renderer.render(scene, camera, flush=False)
    stats.render()
    canvas.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
