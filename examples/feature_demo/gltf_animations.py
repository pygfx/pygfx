"""
Animations with glTF
====================

This example demonstrates how to load a glTF model with animations and play them.

Model from mixamo.com(https://www.mixamo.com)

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

gltf_path = model_dir / "Soldier.glb"

canvas = WgpuCanvas(size=(800, 600), max_fps=-1, title="Animations", vsync=False)

renderer = gfx.WgpuRenderer(canvas)
camera = gfx.PerspectiveCamera(45, 800 / 600, depth_range=(1, 100))
camera.local.position = (1, 2, -3)
camera.look_at((0, 1, 0))
scene = gfx.Scene()

dl = gfx.DirectionalLight()
dl.local.position = (-3, 10, -10)
scene.add(gfx.AmbientLight(), dl)

gltf = gfx.load_gltf(gltf_path, quiet=True)

# gfx.print_tree(gltf.scene) # Uncomment to see the tree structure

model_obj = gltf.scene.children[0]

skeleton_helper = gfx.SkeletonHelper(model_obj)
scene.add(skeleton_helper)
scene.add(model_obj)

camera.show_object(model_obj, view_dir=(-1, -1, 3))

gfx.OrbitController(camera, register_events=renderer)

stats = gfx.Stats(viewport=renderer)

clock = gfx.Clock()

clips = gltf.animations

gui_renderer = ImguiRenderer(renderer.device, canvas)

state = {
    "model": True,
    "skeleton": False,
    "selected_action": 2,
}


def draw_imgui():
    imgui.new_frame()
    imgui.set_next_window_size((250, 0), imgui.Cond_.always)
    imgui.set_next_window_pos(
        (gui_renderer.backend.io.display_size.x - 250, 0), imgui.Cond_.always
    )
    is_expand, _ = imgui.begin(
        "Controls",
        None,
        flags=imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_resize,
    )
    if is_expand:
        if imgui.collapsing_header("Visibility", imgui.TreeNodeFlags_.default_open):
            _, state["model"] = imgui.checkbox("show model", state["model"])
            if state["model"]:
                model_obj.visible = True
            else:
                model_obj.visible = False

            _, state["skeleton"] = imgui.checkbox("show skeleton", state["skeleton"])
            if state["skeleton"]:
                skeleton_helper.visible = True
            else:
                skeleton_helper.visible = False

        if imgui.collapsing_header("Animations", imgui.TreeNodeFlags_.default_open):
            selected, state["selected_action"] = imgui.combo(
                "Animation",
                state["selected_action"],
                [c.name for c in clips],
                len(clips),
            )
            if selected:
                clock.start()  # restart the animation

    imgui.end()
    imgui.end_frame()
    imgui.render()
    return imgui.get_draw_data()


gui_renderer.set_gui(draw_imgui)


def animate():
    elapsed_time = clock.get_elapsed_time()
    clip = clips[state["selected_action"]]

    t = elapsed_time % clip.duration  # simple loop the animation for now
    clip.update(t)

    model_obj.children[0].skeleton.update()
    skeleton_helper.update()

    with stats:
        renderer.render(scene, camera, flush=False)
    stats.render()
    gui_renderer.render()
    canvas.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
