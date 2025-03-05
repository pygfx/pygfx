"""
Facecap animation with morph targets
====================================

This example demonstrates how to animate a model with morph targets.

The model originates from Face Cap (https://www.bannaflak.com/face-cap/documentation.html#1.5)
and has undergone format conversion.
It includes morph targets for facial expressions, utilizing 52 blend shapes.
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

import pygfx as gfx

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

scene.add(model_obj)

camera.show_object(model_obj, view_dir=(1.8, -0.8, -3), scale=1.2)

gfx.OrbitController(camera, register_events=renderer)


stats = gfx.Stats(viewport=renderer)

face_mesh = model_obj.children[0].children[0].children[2].children[0]
gui_renderer = ImguiRenderer(renderer.device, canvas)

mixer = gfx.AnimationMixer()

action_clip = gltf.animations[0]
action = mixer.clip_action(action_clip)

clock = gfx.Clock()

action.play()


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
        for i, name in enumerate(face_mesh.morph_target_names):
            imgui.slider_float(name, face_mesh.morph_target_influences[i], 0, 1)

    imgui.end()
    imgui.end_frame()
    imgui.render()
    return imgui.get_draw_data()


gui_renderer.set_gui(draw_imgui)


def animate():
    dt = clock.get_delta()
    mixer.update(dt)

    with stats:
        renderer.render(scene, camera, flush=False)
    stats.render()
    gui_renderer.render()
    canvas.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
