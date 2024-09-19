"""
Facecap animation with morph targets
====================================

This example demonstrates how to animate a model with morph targets.
The model is a facecap model with morph targets for facial expressions (52 blend shapes).
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

import time
import numpy as np
import pygfx as gfx
from scipy import interpolate
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.utils.imgui import ImguiRenderer
from imgui_bundle import imgui

gltf_path = model_dir / "facecap.glb"

scene = gfx.Scene()

canvas = WgpuCanvas(size=(1280, 720), max_fps=-1, title="Facecap", vsync=False)

renderer = gfx.WgpuRenderer(canvas)
camera = gfx.PerspectiveCamera(45, 1280 / 720, depth_range=(0.1, 1000))

direct_light = gfx.DirectionalLight()
direct_light.local.position = (0, 1, 1)

scene.add(gfx.AmbientLight(), direct_light)

gltf = gfx.load_gltf(gltf_path)

model_obj = gltf.scene.children[1]

action_clip = gltf.animations[0]

scene.add(model_obj)

camera.show_object(model_obj, view_dir=(1.8, -0.8, -3), scale=1.2)

gfx.OrbitController(camera, register_events=renderer)


def update_track(track, time):
    target = track["target"]
    property = track["property"]
    values = track["values"]
    times = track["times"]
    interpolation = track["interpolation"]

    if time < times[0]:
        time = times[0]

    values = values.reshape(len(times), -1)

    # TODO: Use scipy to interpolate now, will use our own implementation later
    if interpolation == "LINEAR":
        if property == "rotation":
            # TODO: should use spherical linear interpolation instead
            cs = interpolate.interp1d(times, values, kind="linear", axis=0)
            value = cs(time)
            value = value / np.linalg.norm(value)  # normalize quaternion
        else:
            cs = interpolate.interp1d(times, values, kind="linear", axis=0)
            value = cs(time)

    elif interpolation == "CUBICSPLINE":
        cs = interpolate.interp1d(times, values, kind="cubic", axis=0)
        value = cs(time)
    elif interpolation == "STEP":
        cs = interpolate.interp1d(times, values, kind="previous", axis=0)
        value = cs(time)
    else:
        print("unknown interpolation", interpolation)

    if property == "scale":
        target.local.scale = value
    elif property == "translation":
        target.local.position = value
    elif property == "rotation":
        target.local.rotation = value
    elif property == "weights":
        target.morph_target_influences = value
        # target.morph_target_influences = np.ones_like(value)
    else:
        print("unknown property", property)


tracks = action_clip["tracks"]
gloabl_time = 0
last_time = time.perf_counter()

stats = gfx.Stats(viewport=renderer)

face_mesh = model_obj.children[0].children[0].children[2].children[0]
gui_renderer = ImguiRenderer(renderer.device, canvas)


def draw_imgui():
    imgui.new_frame()
    imgui.set_next_window_size((400, 0), imgui.Cond_.always)
    imgui.set_next_window_pos(
        (gui_renderer.backend.io.display_size.x - 400, 0), imgui.Cond_.always
    )
    imgui.set_next_item_open(True)
    is_expand, _ = imgui.begin(
        "Controls",
        None,
        flags=imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_resize,
    )
    if is_expand:
        # imgui.begin_disabled()
        for i, name in enumerate(face_mesh.morph_target_names):
            imgui.slider_float(name, face_mesh.morph_target_influences[i], 0, 1)
        # imgui.end_disabled()

    imgui.end()
    imgui.end_frame()
    imgui.render()
    return imgui.get_draw_data()


gui_renderer.set_gui(draw_imgui)


def animate():
    global gloabl_time, last_time
    now = time.perf_counter()
    dt = now - last_time
    last_time = now
    gloabl_time += dt
    if gloabl_time > action_clip["duration"]:
        gloabl_time = 0

    for track in tracks:
        update_track(track, gloabl_time)

    with stats:
        renderer.render(scene, camera, flush=False)
    stats.render()
    gui_renderer.render()
    canvas.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
