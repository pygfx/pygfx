"""
Remote glTF Viewer
==================

This example demonstrates loading glTF models from remote URLs (KhronosGroup glTF-Sample-Assets).

"""

# sphinx_gallery_pygfx_docs = 'animate 4s'
# sphinx_gallery_pygfx_test = 'run'

import pygfx as gfx
from wgpu.gui.auto import WgpuCanvas, run

from wgpu.utils.imgui import ImguiRenderer
from imgui_bundle import imgui, imspinner  # type: ignore

import httpx
import threading

canvas = WgpuCanvas(size=(1280, 720), max_fps=-1, title="glTF viewer", vsync=False)

renderer = gfx.WgpuRenderer(canvas)

scene = gfx.Scene()

scene.add(gfx.Background.from_color(0.1))

dl = gfx.DirectionalLight()
dl.local.position = (-3, 10, -10)
scene.add(gfx.AmbientLight(intensity=0.5), dl)
camera = gfx.PerspectiveCamera(45, 1280 / 720)

gfx.OrbitController(camera, register_events=renderer)

stats = gfx.Stats(viewport=renderer)

clock = gfx.Clock()

gui_renderer = ImguiRenderer(renderer.device, canvas)

state = {
    "model": True,
    "skeleton": False,
    "selected_model": 0,
    "animate": True,
    "selected_action": 0,
    "loading": False,
}

base_url = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main"

model_list_url = f"{base_url}/Models/model-index.json"

response = httpx.get(model_list_url, follow_redirects=True)
response.raise_for_status()

model_list: list = response.json()

# filter out models having "core" in tags
model_list = [m for m in model_list if "core" in m.get("tags", [])]

mixer = gfx.AnimationMixer()

model_obj = None
skeleton_helper = None
actions = None

favorite_variant = ["glTF-Binary", "glTF-Embedded", "glTF"]


def load_model(model_index):
    global model_obj, skeleton_helper, actions

    model_folder = base_url + "/Models"
    # model_index = state["selected_model"]
    model_desc = model_list[model_index]
    if model_desc.get("name", None) and model_desc.get("variants", None):
        for variant in favorite_variant:
            if variant in model_desc["variants"]:
                model_path = (
                    model_folder
                    + f"/{model_desc["name"]}/{variant}/{model_desc["variants"][variant]}"
                )
                print("Loading model", model_path)

                state["loading"] = True
                try:
                    gltf = gfx.load_gltf(model_path)

                    if model_obj:
                        scene.remove(model_obj)
                    if skeleton_helper:
                        scene.remove(skeleton_helper)

                    model_obj = gltf.scene if gltf.scene else gltf.scenes[0]
                    skeleton_helper = gfx.SkeletonHelper(model_obj)
                    scene.add(skeleton_helper)
                    scene.add(model_obj)
                    state["selected_action"] = 0

                    camera.show_object(model_obj, scale=1.4)
                    actions = None
                    clips = gltf.animations
                    if clips:
                        actions = [mixer.clip_action(clip) for clip in clips]
                        if state["animate"]:
                            actions[state["selected_action"]].play()
                except Exception as e:
                    print(e)
                state["loading"] = False
                return

    print("Model not found", model_desc)


def draw_imgui():
    global model_obj, skeleton_helper, actions
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
        if imgui.collapsing_header("Models", imgui.TreeNodeFlags_.default_open):
            if state["loading"]:
                imgui.begin_disabled()

            selected, state["selected_model"] = imgui.combo(
                " ",
                state["selected_model"],
                [m["name"] for m in model_list],
                10,
            )

            if selected:
                threading.Thread(
                    target=load_model, args=(state["selected_model"],)
                ).start()

            if state["loading"]:
                imgui.end_disabled()
                imgui.same_line()
                imspinner.spinner_arc_rotation(
                    "loading", 8, 4.0, imgui.ImColor(0.3, 0.5, 0.9, 1.0), speed=1.0
                )

        if imgui.collapsing_header("Visibility", imgui.TreeNodeFlags_.default_open):
            if model_obj:
                _, state["model"] = imgui.checkbox("show model", state["model"])
                if state["model"]:
                    model_obj.visible = True
                else:
                    model_obj.visible = False

            if skeleton_helper:
                _, state["skeleton"] = imgui.checkbox(
                    "show skeleton", state["skeleton"]
                )
                if state["skeleton"]:
                    skeleton_helper.visible = True
                else:
                    skeleton_helper.visible = False

        if actions:
            if imgui.collapsing_header("Animations", imgui.TreeNodeFlags_.default_open):
                changed, state["animate"] = imgui.checkbox("Animate", state["animate"])
                if changed:
                    for action in actions:
                        action.stop()
                    if state["animate"]:
                        actions[state["selected_action"]].play()

                selected, state["selected_action"] = imgui.combo(
                    "Animation",
                    state["selected_action"],
                    [a._clip.name if a._clip.name else "unnamed" for a in actions],
                    len(actions),
                )
                if selected:
                    for action in actions:
                        action.stop()
                    if state["animate"]:
                        actions[state["selected_action"]].play()

    imgui.end()
    imgui.end_frame()
    imgui.render()
    return imgui.get_draw_data()


gui_renderer.set_gui(draw_imgui)

load_model(0)


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
