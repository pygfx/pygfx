"""
Skinning Animation
==================

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
import pygfx as gfx
import pylinalg as la
from wgpu.gui.auto import WgpuCanvas, run
from  pygfx.utils.gltf_loader import GLTF, print_tree

gltf_path = model_dir / "Michelle.glb"

scene = gfx.Scene()

canvas = WgpuCanvas(size=(640, 480), max_fps=60, title="Skinnedmesh")

renderer = gfx.WgpuRenderer(canvas)
camera = gfx.PerspectiveCamera(75, 640 / 480, depth_range=(0.1, 1000))
camera.local.position = (0, 100, 200)
camera.look_at((0, 100, 0))
scene = gfx.Scene()

scene.add(gfx.AmbientLight(), gfx.DirectionalLight())


gltf = GLTF.load(gltf_path)
# print_tree(gltf.scene)
mesh_obj = gltf.scene.children[0]

mesh_obj.local.scale = ( 1, 1, 1 )

action_clip = gltf.animations[0]

print("action_clip", action_clip["name"], action_clip["duration"], len(action_clip["tracks"]))

skeleton_helper = gfx.SkeletonHelper(gltf.scene)
scene.add(skeleton_helper)

scene.add(gltf.scene)

# xyz = gfx.AxesHelper( 20 )
# scene.add( xyz )

gfx.OrbitController(camera, register_events=renderer)


def update_track(track, time):
    from scipy import interpolate
    import numpy as np

    target = track["target"]
    property = track["property"]
    values = track["values"]
    times = track["times"]
    interpolation = track["interpolation"]

    if time < times[0]:
        time = times[0]

    if interpolation == "LINEAR":
        cs = interpolate.interp1d(times, values, kind='linear', axis=0)
        value = cs(time)
    elif interpolation == "CUBICSPLINE":
        cs = interpolate.CubicSpline(times, values, bc_type='natural')
        value = cs(time)
    elif interpolation == "STEP":
        for i in range(len(times)):
            if time < times[i]:
                break
        if i == 0:
            i = 1
        v0 = values[i - 1]
        value = v0
    else:
        print("unknown interpolation", interpolation)

    # if property == "scale":
    #     target.local.scale = value
    # elif property == "translation":
    #     target.local.position = value
    # elif property == "rotation":
    #     # TODO: should use spherical linear interpolation instead
    #     target.local.rotation = value
    # elif property == "quaternion":
    #     target.local.quaternion = value / np.linalg.norm(value)  # normalize quaternion
    # else:
    #     print("unknown property", property)

    target_temp = getattr(target, "_temp", None)

    if target_temp is None:
        target._temp = {}
        target_temp = target._temp

    if property == "rotation":
        value = value / np.linalg.norm(value)  # normalize quaternion

    target_temp[property] = value

    
    if "scale" in target_temp and "translation" in target_temp and "rotation" in target_temp:
        matrix = la.mat_compose(target_temp["translation"], target_temp["rotation"], target_temp["scale"])
        target.local.matrix = matrix
        del target._temp


tracks = action_clip["tracks"]
gloabl_time = 0
last_time = time.time()

stats = gfx.Stats(viewport=renderer)

def animate():
    global gloabl_time, last_time
    now = time.time()
    dt = now - last_time
    last_time = now
    gloabl_time += dt
    if gloabl_time > action_clip["duration"]:
        gloabl_time = 0

    for track in tracks:
        update_track(track, gloabl_time)

    mesh_obj.children[0].skeleton.update()
    skeleton_helper.update()

    with stats:
        renderer.render(scene, camera, flush=False)
    stats.render()
    canvas.request_draw()

if __name__ == "__main__":
    renderer.request_draw(animate)
    run()