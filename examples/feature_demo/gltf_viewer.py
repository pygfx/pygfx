"""
Remote glTF Viewer
==================

This example demonstrates loading glTF models from remote URLs (KhronosGroup glTF-Sample-Assets).

"""

# sphinx_gallery_pygfx_docs = 'code'
# sphinx_gallery_pygfx_test = 'off'

import pygfx as gfx
from rendercanvas.glfw import RenderCanvas, loop
import imageio.v3 as iio

from wgpu.utils.imgui import ImguiRenderer, Stats
from imgui_bundle import imgui, imspinner  # type: ignore
from imgui_bundle import portable_file_dialogs as pfd  # type: ignore
from imgui_bundle import hello_imgui, icons_fontawesome_6  # type: ignore
from pygfx.utils.text import FontProps

import httpx
import threading
import asyncio
import glfw

canvas = RenderCanvas(
    size=(1280, 720), update_mode="fastest", title="glTF viewer", vsync=False
)


renderer = gfx.WgpuRenderer(canvas)
glfw.maximize_window(canvas._window)
glfw.set_window_aspect_ratio(canvas._window, 16, 9)

scene = gfx.Scene()


ambient_light = gfx.AmbientLight(intensity=0.3)
scene.add(ambient_light)
directional_light = gfx.DirectionalLight(intensity=2.5)
directional_light.local.position = (0.5, 0, 0.866)
scene.add(directional_light)

camera = gfx.PerspectiveCamera(45, 1280 / 720)

gfx.OrbitController(camera, register_events=renderer)

stats = Stats(device=renderer.device, canvas=canvas)

clock = gfx.Clock()

gui_renderer = ImguiRenderer(renderer.device, canvas)

fonts = gfx.font_manager.select_font("Controls", FontProps())
fonts_file = fonts[0][1]._filename
font = gui_renderer.backend.io.fonts.add_font_from_file_ttf(fonts_file, 16)

gui_renderer.backend.io.font_default = font

fa_loading_params = hello_imgui.FontLoadingParams(merge_to_last_font=True)
fa = hello_imgui.load_font("fonts/fontawesome-webfont.ttf", 14, fa_loading_params)

state = {
    "selected_model": 0,
    "animate": True,
    "selected_action": 0,
    "loading": False,
    "ibl": True,
    "show_inspector": False,
    "current_inspector_target": None,
}

base_url = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main"

model_list_url = f"{base_url}/Models/model-index.json"

response = httpx.get(model_list_url, follow_redirects=True)
response.raise_for_status()

model_list: list = response.json()

# filter out models having "core" in tags
# model_list = [m for m in model_list if "core" in m.get("tags", [])]

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

scene.add(gfx.Background.from_color((0.1, 0.1, 0.1, 1)))


def add_env_map(obj, env_map):
    if isinstance(obj, gfx.Mesh) and isinstance(obj.material, gfx.MeshStandardMaterial):
        obj.material.env_map = env_map


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

inspector_materials = []


def collect_inspector_targets(obj):
    inspector_materials.clear()

    def visit(obj):
        if (
            hasattr(obj, "material")
            and obj.material is not None
            and obj.material not in inspector_materials
        ):
            inspector_materials.append(obj.material)

    obj.traverse(visit)
    return inspector_materials


def load_model(model_path):
    global model_obj, skeleton_helper, actions
    try:
        gltf = asyncio.run(gfx.load_gltf_async(model_path))

        if model_obj:
            scene.remove(model_obj)
        if skeleton_helper:
            scene.remove(skeleton_helper)

        model_obj = gltf.scene if gltf.scene else gltf.scenes[0]
        if state["ibl"]:
            model_obj.traverse(lambda obj: add_env_map(obj, env_tex))

        skeleton_helper = gfx.SkeletonHelper(model_obj)
        skeleton_helper.visible = False
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

        collect_inspector_targets(model_obj)
    except Exception as e:
        print(e)


def draw_imgui():
    global model_obj, skeleton_helper, actions, open_file_dialog

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

        if imgui.collapsing_header("Lighting", imgui.TreeNodeFlags_.default_open):
            _, ambient_light.visible = imgui.checkbox(
                "Ambient Light", ambient_light.visible
            )

            _, directional_light.visible = imgui.checkbox(
                "Directional Light", directional_light.visible
            )

            changed, state["ibl"] = imgui.checkbox("IBL", state["ibl"])
            if changed:
                if state["ibl"]:
                    model_obj.traverse(lambda obj: add_env_map(obj, env_tex))
                else:
                    model_obj.traverse(lambda obj: add_env_map(obj, None))

        if imgui.collapsing_header("Visibility", imgui.TreeNodeFlags_.default_open):
            _, background.visible = imgui.checkbox(
                "show background", background.visible
            )

            if model_obj:
                _, model_obj.visible = imgui.checkbox("show model", model_obj.visible)

            if skeleton_helper:
                _, skeleton_helper.visible = imgui.checkbox(
                    "show skeleton", skeleton_helper.visible
                )

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

        if model_obj:
            if imgui.collapsing_header("Inspector", imgui.TreeNodeFlags_.default_open):
                if imgui.button(
                    "Show Inspector",
                ):
                    state["show_inspector"] = not state["show_inspector"]
    imgui.end()

    if state["show_inspector"]:
        show_inspector()


def show_inspector():
    imgui.set_next_window_size((600, 600), imgui.Cond_.appearing)

    _, state["show_inspector"] = imgui.begin(
        "Inspector",
        state["show_inspector"],
    )

    if imgui.begin_child(
        "##tree", (300, 0), imgui.ChildFlags_.resize_x | imgui.ChildFlags_.borders
    ):
        if model_obj:
            if imgui.tree_node("Nodes"):

                def make_tree_node(obj):
                    imgui.push_id(str(id(obj)))
                    node_flags = (
                        imgui.TreeNodeFlags_.draw_lines_to_nodes
                        | imgui.TreeNodeFlags_.open_on_arrow
                        | imgui.TreeNodeFlags_.span_full_width
                        | imgui.TreeNodeFlags_.allow_overlap
                    )

                    if not obj.children:
                        node_flags |= imgui.TreeNodeFlags_.leaf

                    if state["current_inspector_target"] == obj:
                        node_flags |= imgui.TreeNodeFlags_.selected

                    node_open = imgui.tree_node_ex("", node_flags)

                    if imgui.is_item_clicked():
                        state["current_inspector_target"] = obj

                    icon = icons_fontawesome_6.ICON_FA_CODE_BRANCH
                    icon_color = imgui.IM_COL32(100, 149, 237, 255)
                    if isinstance(obj, gfx.Mesh):
                        icon = icons_fontawesome_6.ICON_FA_CUBE
                        icon_color = imgui.IM_COL32(30, 144, 255, 255)
                    elif isinstance(obj, gfx.Light):
                        icon = icons_fontawesome_6.ICON_FA_SUN
                        icon_color = imgui.IM_COL32(255, 215, 0, 255)
                    elif isinstance(obj, gfx.Camera):
                        icon = icons_fontawesome_6.ICON_FA_CAMERA
                        icon_color = imgui.IM_COL32(255, 105, 180, 255)
                    elif isinstance(obj, gfx.Bone):
                        icon = icons_fontawesome_6.ICON_FA_GEAR
                        icon_color = imgui.IM_COL32(255, 255, 255, 255)

                    imgui.push_style_color(imgui.Col_.text, icon_color)
                    imgui.same_line(spacing=1)
                    imgui.text(icon)
                    imgui.pop_style_color()

                    imgui.same_line(spacing=5)
                    imgui.text(obj.name)

                    if isinstance(obj, gfx.Mesh):
                        imgui.same_line()

                        imgui.set_cursor_pos_x(
                            imgui.get_cursor_pos_x()
                            + imgui.get_content_region_avail().x
                            - 24
                        )

                        imgui.push_style_color(imgui.Col_.button, (0, 0, 0, 0))
                        imgui.push_style_color(
                            imgui.Col_.button_hovered, (0.2, 0.2, 0.2, 0.5)
                        )
                        imgui.push_style_color(
                            imgui.Col_.button_active, (0.3, 0.3, 0.3, 0.5)
                        )

                        if obj.visible:
                            imgui.push_style_color(imgui.Col_.text, (1, 1, 1, 1))
                        else:
                            imgui.push_style_color(imgui.Col_.text, (0.5, 0.5, 0.5, 1))

                        if imgui.small_button(
                            icons_fontawesome_6.ICON_FA_EYE
                            if obj.visible
                            else icons_fontawesome_6.ICON_FA_EYE_SLASH
                        ):
                            obj.visible = not obj.visible

                        imgui.pop_style_color(4)

                        if imgui.is_item_hovered():
                            imgui.set_tooltip("Toggle visibility")

                    imgui.pop_id()

                    if node_open:
                        for child in obj.children:
                            make_tree_node(child)

                        imgui.tree_pop()

                make_tree_node(model_obj)
                imgui.tree_pop()

            if imgui.tree_node("Materials"):
                if inspector_materials:
                    for mat in inspector_materials:
                        imgui.push_id(str(id(mat)))
                        imgui.push_style_color(
                            imgui.Col_.text, imgui.IM_COL32(255, 165, 0, 255)
                        )
                        _, clicked = imgui.selectable(
                            f"{icons_fontawesome_6.ICON_FA_SUITCASE}",
                            state["current_inspector_target"] == mat,
                            imgui.SelectableFlags_.span_all_columns,
                        )
                        imgui.pop_style_color()
                        if clicked:
                            state["current_inspector_target"] = mat

                        imgui.same_line(spacing=10)
                        imgui.text(mat.name)
                        imgui.pop_id()

                imgui.tree_pop()

    imgui.end_child()

    imgui.same_line()

    # Inspector details
    imgui.begin_group()
    if state["current_inspector_target"]:
        target = state["current_inspector_target"]

        # Title
        imgui.push_style_color(imgui.Col_.text, imgui.IM_COL32(30, 144, 255, 255))
        imgui.push_font(font, 20.0)
        imgui.text(f"{target.name}")
        imgui.pop_font()
        imgui.pop_style_color()

        imgui.text_disabled(f"Type: {type(target).__name__}")

        imgui.separator()
        # Show properties based on type
        imgui.dummy((0, 10))
        if isinstance(target, gfx.Material):
            show_material_properties(target)
        elif isinstance(target, gfx.WorldObject):
            show_world_object_properties(target)

    imgui.end_group()
    imgui.end()


def show_material_properties(material):
    transparency_mode = 0

    if material.blending["name"] == "dither":
        transparency_mode = 3
    elif material.transparent is True:
        transparency_mode = 2
    elif material.alpha_test > 0:
        transparency_mode = 1
    else:
        transparency_mode = 0

    imgui.text("Transparency Mode")
    imgui.same_line()
    imgui.set_next_item_width(150)
    selected, transparency_mode = imgui.combo(
        "##",
        transparency_mode,
        ["Opaque", "Mask", "Blend", "Dither"],
    )
    if selected:
        if transparency_mode == 0:
            material.transparent = False
            material.blending = "normal"
            material.alpha_test = 0
        elif transparency_mode == 1:
            material.transparent = False
            material.blending = "normal"
            material.alpha_test = 0.5
        elif transparency_mode == 2:
            material.transparent = True
            material.blending = "normal"
            material.alpha_test = 0
        elif transparency_mode == 3:
            material.transparent = True
            material.blending = "dither"
            material.alpha_test = 0

    _, material.depth_write = imgui.checkbox("Depth Write", material.depth_write)

    _, material.depth_test = imgui.checkbox("Depth Test", material.depth_test)


def show_world_object_properties(obj):
    position = list(obj.local.position)
    rotation = list(obj.local.rotation)
    scale = list(obj.local.scale)

    imgui.separator_text("Transform")
    imgui.begin_table("Transform", 2)
    imgui.table_next_row()
    imgui.table_next_column()
    imgui.text("Position")
    imgui.table_next_column()
    imgui.text(f"{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}")
    imgui.table_next_row()
    imgui.table_next_column()
    imgui.text("Rotation")
    imgui.table_next_column()
    imgui.text(
        f"{rotation[0]:.2f}, {rotation[1]:.2f}, {rotation[2]:.2f}, {rotation[3]:.2f}"
    )
    imgui.table_next_row()
    imgui.table_next_column()
    imgui.text("Scale")
    imgui.table_next_column()
    imgui.text(f"{scale[0]:.2f}, {scale[1]:.2f}, {scale[2]:.2f}")
    imgui.end_table()

    if isinstance(obj, gfx.Mesh):
        imgui.separator_text("Geometry")
        vertices = obj.geometry.positions.nitems
        faces = obj.geometry.indices.nitems
        imgui.text("Vertices:")
        imgui.same_line()
        imgui.text(str(vertices))
        imgui.text("Faces:")
        imgui.same_line()
        imgui.text(str(faces))


gui_renderer.set_gui(draw_imgui)
# load_remote_model(0)


def animate():
    dt = clock.get_delta()
    mixer.update(dt)

    with stats:
        renderer.render(scene, camera)

    gui_renderer.render()
    canvas.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    loop.run()
