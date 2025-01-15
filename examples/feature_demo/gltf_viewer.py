"""
Remote glTF Viewer
==================

This example demonstrates loading glTF models from remote URLs (KhronosGroup glTF-Sample-Assets).

"""

# sphinx_gallery_pygfx_docs = 'code'
# sphinx_gallery_pygfx_test = 'off'

import pygfx as gfx
from wgpu.gui.auto import WgpuCanvas, run
import imageio.v3 as iio

from wgpu.utils.imgui import ImguiRenderer
from imgui_bundle import imgui, imspinner  # type: ignore
from imgui_bundle import portable_file_dialogs as pfd  # type: ignore

import httpx
import threading
import asyncio

canvas = WgpuCanvas(size=(1280, 720), max_fps=-1, title="glTF viewer", vsync=False)

renderer = gfx.WgpuRenderer(canvas)

scene = gfx.Scene()

dl = gfx.DirectionalLight()
dl.local.position = (-3, 10, 10)
scene.add(gfx.AmbientLight(intensity=0.1), dl)
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
    "background": False,
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

# Read cube image and turn it into a 3D image (a 4d array)
env_img = iio.imread("imageio:meadow_cube.jpg")
cube_size = env_img.shape[1]
env_img.shape = 6, cube_size, cube_size, env_img.shape[-1]

# Create environment map
env_tex = gfx.Texture(
    env_img, dim=2, size=(cube_size, cube_size, 6), generate_mipmaps=True
)

background = gfx.Background(None, gfx.BackgroundSkyboxMaterial(map=env_tex))
background.visible = False
scene.add(background)


def add_env_map(obj):
    if isinstance(obj, gfx.Mesh) and isinstance(obj.material, gfx.MeshStandardMaterial):
        obj.material.env_map = env_tex


def load_remote_model(model_index):
    model_folder = base_url + "/Models"
    model_desc = model_list[model_index]
    if model_desc.get("name", None) and model_desc.get("variants", None):
        for variant in favorite_variant:
            if variant in model_desc["variants"]:
                model_path = (
                    model_folder
                    + f"/{model_desc['name']}/{variant}/{model_desc['variants'][variant]}"
                )
                print("Loading model", model_path)

                state["loading"] = True
                load_model(model_path)
                state["loading"] = False
                return

    print("Model not found", model_desc)


open_file_dialog = None


def load_model(model_path):
    global model_obj, skeleton_helper, actions
    try:
        gltf = asyncio.run(gfx.load_gltf_async(model_path))

        if model_obj:
            scene.remove(model_obj)
        if skeleton_helper:
            scene.remove(skeleton_helper)

        model_obj = gltf.scene if gltf.scene else gltf.scenes[0]
        model_obj.traverse(add_env_map)

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


def draw_imgui():
    global model_obj, skeleton_helper, actions, open_file_dialog
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
            imgui.begin_disabled(state["loading"])

            selected, state["selected_model"] = imgui.combo(
                " ",
                state["selected_model"],
                [m["name"] for m in model_list],
                10,
            )

            if selected:
                threading.Thread(
                    target=load_remote_model, args=(state["selected_model"],)
                ).start()

            imgui.end_disabled()
            if state["loading"]:
                imgui.same_line()
                imspinner.spinner_arc_rotation(
                    "loading", 8, 4.0, imgui.ImColor(0.3, 0.5, 0.9, 1.0), speed=1.0
                )

            if imgui.button("Open local model file"):
                open_file_dialog = pfd.open_file(
                    "Select file", ".", filters=["*.gltf, *.glb", "*.gltf *.glb"]
                )

            if open_file_dialog is not None and open_file_dialog.ready():
                files = open_file_dialog.result()
                if files:
                    threading.Thread(target=load_model, args=(files[0],)).start()
                open_file_dialog = None

        if imgui.collapsing_header("Visibility", imgui.TreeNodeFlags_.default_open):
            _, state["background"] = imgui.checkbox(
                "show background", state["background"]
            )
            if state["background"]:
                background.visible = True
            else:
                background.visible = False

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

load_remote_model(0)


def update_skeleton(obj):
    if isinstance(obj, gfx.SkinnedMesh):
        obj.skeleton.update()


def animate():
    dt = clock.get_delta()
    mixer.update(dt)

    # Update the skeleton
    # todo: this is a temporary workaround, we should update the skeleton automatically
    # see: https://github.com/pygfx/pygfx/pull/894
    if skeleton_helper and skeleton_helper.visible:
        skeleton_helper.update()
    if model_obj:
        model_obj.traverse(update_skeleton)

    with stats:
        renderer.render(scene, camera, flush=False)
    stats.render()
    gui_renderer.render()
    canvas.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
