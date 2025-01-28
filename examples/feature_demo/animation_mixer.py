"""
Animation Mixer
===============

This example demonstrates how to use the AnimationMixer to blend between different animations.

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
from pygfx.utils.text import FontProps
from wgpu.gui.auto import WgpuCanvas, run

from wgpu.utils.imgui import ImguiRenderer
from imgui_bundle import imgui, hello_imgui  # type: ignore

gltf_path = model_dir / "Soldier.glb"

canvas = WgpuCanvas(size=(1280, 720), max_fps=-1, title="Animation Mixer", vsync=False)

renderer = gfx.WgpuRenderer(canvas)
camera = gfx.PerspectiveCamera(45, 1280 / 720, depth_range=(1, 100))
camera.local.position = (1, 2, -4)
camera.look_at((0, 1, 0))
scene = gfx.Scene()

dl = gfx.DirectionalLight()
dl.local.position = (-3, 10, -10)
scene.add(gfx.AmbientLight(), dl)


gltf = gfx.load_gltf(gltf_path, quiet=True)

# gfx.print_scene_graph(gltf.scene) # Uncomment to see the tree structure

model_obj = gltf.scene.children[0]

action_clip = gltf.animations[0]

skeleton_helper = gfx.SkeletonHelper(model_obj)
scene.add(skeleton_helper)
scene.add(model_obj)

gfx.OrbitController(camera, register_events=renderer)

stats = gfx.Stats(viewport=renderer)

clock = gfx.Clock()

animation_mixer = gfx.AnimationMixer()

idle_action = animation_mixer.clip_action(gltf.animations[0])
walk_action = animation_mixer.clip_action(gltf.animations[3])
run_action = animation_mixer.clip_action(gltf.animations[1])

actions = [idle_action, walk_action, run_action]


gui_renderer = ImguiRenderer(renderer.device, canvas)

fonts = gfx.font_manager.select_font("Controls", FontProps())
fonts_file = fonts[0][1]._filename
font = gui_renderer.backend.io.fonts.add_font_from_file_ttf(fonts_file, 16)
gui_renderer.backend.create_fonts_texture()

gui_renderer.backend.io.font_default = font

state = {
    "model": True,
    "skeleton": False,
    "pause": False,
    "walk_to_idle": False,
    "idle_to_walk": False,
    "walk_to_run": False,
    "run_to_walk": False,
    "use_custom_duration": False,
    "custom_duration": 3.5,
    "single_step_mode": False,
    "next_step_size": 0,
    "next_step_size_config": 0.05,
    "idle_weight": 0,
    "walk_weight": 1,
    "run_weight": 0,
}


def set_weight(action, weight):
    action.enabled = True
    action.set_effective_time_scale(1.0)
    action.set_effective_weight(weight)


def deactivate_all():
    for a in actions:
        a.stop()


def activate_all():
    set_weight(idle_action, state["idle_weight"])
    set_weight(walk_action, state["walk_weight"])
    set_weight(run_action, state["run_weight"])
    for a in actions:
        a.play()


activate_all()


def sync_cross_fade(action1, action2, duration):
    def on_loop(event):
        if getattr(event, "action", None) == action1:
            animation_mixer.remove_event_handler(on_loop, "loop")
            cross_fade(action1, action2, duration)

    animation_mixer.add_event_handler(on_loop, "loop")


def cross_fade(start_action, end_action, duration):
    state["single_step_mode"] = False
    unpause_all()

    duration = state["custom_duration"] if state["use_custom_duration"] else duration

    set_weight(end_action, 1)
    end_action.time = 0
    start_action.cross_fade_to(end_action, duration, True)


def pause_all():
    state["pause"] = True
    for a in actions:
        a.paused = True


def unpause_all():
    state["pause"] = False
    for a in actions:
        a.paused = False


def to_single_step_mode():
    unpause_all()
    state["single_step_mode"] = True
    state["next_step_size"] = state["next_step_size_config"]


tweaked_theme = hello_imgui.ImGuiTweakedTheme()
tweaked_theme.theme = hello_imgui.ImGuiTheme_.photoshop_style
tweaked_theme.tweaks.rounding = 0.0
hello_imgui.push_tweaked_theme(tweaked_theme)

next_step_size = 0.05


def draw_imgui():
    imgui.new_frame()
    imgui.set_next_window_size((300, 0), imgui.Cond_.always)
    imgui.set_next_window_pos(
        (gui_renderer.backend.io.display_size.x - 300, 0), imgui.Cond_.always
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

        if imgui.collapsing_header(
            "Activation/Deactivation", imgui.TreeNodeFlags_.default_open
        ):
            if imgui.button("deactivate all", size=(-1, 0)):
                deactivate_all()
            if imgui.button("activate all", size=(-1, 0)):
                activate_all()

        if imgui.collapsing_header(
            "Pausing/Stepping", imgui.TreeNodeFlags_.default_open
        ):
            if imgui.button("pause/continue", size=(-1, 0)):
                if state["single_step_mode"]:
                    state["single_step_mode"] = False
                    unpause_all()
                else:
                    state["pause"] = not state["pause"]
                    for a in actions:
                        a.paused = state["pause"]

            if imgui.button("make single step", size=(-1, 0)):
                to_single_step_mode()

            _, state["next_step_size_config"] = imgui.slider_float(
                "Step Size", state["next_step_size_config"], 0.01, 0.1
            )

        if imgui.collapsing_header("Crossfading", imgui.TreeNodeFlags_.default_open):
            imgui.begin_disabled(not state["walk_to_idle"])
            if imgui.button("from walk to idle", size=(-1, 0)):
                cross_fade(walk_action, idle_action, 1.0)
            imgui.end_disabled()

            imgui.begin_disabled(not state["idle_to_walk"])
            if imgui.button("from idle to walk", size=(-1, 0)):
                cross_fade(idle_action, walk_action, 0.5)
            imgui.end_disabled()

            imgui.begin_disabled(not state["walk_to_run"])
            if imgui.button("from walk to run", size=(-1, 0)):
                sync_cross_fade(walk_action, run_action, 2.5)
            imgui.end_disabled()

            imgui.begin_disabled(not state["run_to_walk"])
            if imgui.button("from run to walk", size=(-1, 0)):
                sync_cross_fade(run_action, walk_action, 5.0)
            imgui.end_disabled()

            _, state["use_custom_duration"] = imgui.checkbox(
                "Use custom duration", state["use_custom_duration"]
            )
            if state["use_custom_duration"]:
                _, state["custom_duration"] = imgui.slider_float(
                    "Duration", state["custom_duration"], 0, 10.0
                )

        if imgui.collapsing_header("Blend Weights", imgui.TreeNodeFlags_.default_open):
            changed, state["idle_weight"] = imgui.slider_float(
                "Idle", idle_action.effective_weight, 0, 1
            )
            if changed:
                set_weight(idle_action, state["idle_weight"])

            changed, state["walk_weight"] = imgui.slider_float(
                "Walk", walk_action.effective_weight, 0, 1
            )
            if changed:
                set_weight(walk_action, state["walk_weight"])

            changed, state["run_weight"] = imgui.slider_float(
                "Run", run_action.effective_weight, 0, 1
            )
            if changed:
                set_weight(run_action, state["run_weight"])

        if imgui.collapsing_header("General Speed", imgui.TreeNodeFlags_.default_open):
            changed, speed = imgui.slider_float(
                "Speed", animation_mixer.time_scale, 0, 1.5
            )
            if changed:
                animation_mixer.time_scale = speed

    imgui.end()
    imgui.end_frame()
    imgui.render()
    return imgui.get_draw_data()


gui_renderer.set_gui(draw_imgui)


def update_cross_fade_controls():
    if (
        idle_action.effective_weight == 1
        and walk_action.effective_weight == 0
        and run_action.effective_weight == 0
    ):
        state["walk_to_idle"] = False
        state["idle_to_walk"] = True
        state["walk_to_run"] = False
        state["run_to_walk"] = False

    if (
        idle_action.effective_weight == 0
        and walk_action.effective_weight == 1
        and run_action.effective_weight == 0
    ):
        state["walk_to_idle"] = True
        state["idle_to_walk"] = False
        state["walk_to_run"] = True
        state["run_to_walk"] = False

    if (
        idle_action.effective_weight == 0
        and walk_action.effective_weight == 0
        and run_action.effective_weight == 1
    ):
        state["walk_to_idle"] = False
        state["idle_to_walk"] = False
        state["walk_to_run"] = False
        state["run_to_walk"] = True


def animate():
    update_cross_fade_controls()

    dt = clock.get_delta()

    if state["single_step_mode"]:
        dt = state["next_step_size"]
        state["next_step_size"] = 0

    animation_mixer.update(dt)

    with stats:
        renderer.render(scene, camera, flush=False)

    stats.render()
    gui_renderer.render()
    canvas.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
