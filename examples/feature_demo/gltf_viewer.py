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
import inspect

from wgpu.utils.imgui import ImguiRenderer, Stats
from imgui_bundle import imgui, imspinner  # type: ignore
from imgui_bundle import portable_file_dialogs as pfd  # type: ignore
from imgui_bundle import hello_imgui, icons_fontawesome_6  # type: ignore

import httpx
import threading
import asyncio
import glfw

import os
from pathlib import Path

try:
    # modify this line if your model is located elsewhere
    assets_dir = Path(__file__).parents[1] / "data"
except NameError:
    # compatibility with sphinx-gallery
    assets_dir = Path(os.getcwd()).parent / "data"

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

fonts = gfx.font_manager.select_font(" ", gfx.font_manager.default_font_props)
fonts_file = fonts[0][1].filename
font = gui_renderer.backend.io.fonts.add_font_from_file_ttf(fonts_file, 16)

gui_renderer.backend.io.font_default = font

fa_loading_params = hello_imgui.FontLoadingParams(
    merge_to_last_font=True, inside_assets=False
)
hello_imgui.load_font(
    str(assets_dir / "Font_Awesome_6_Free-Regular-400.otf"), 14, fa_loading_params
)
fa_loading_params.inside_assets = True
hello_imgui.load_font("fonts/Font_Awesome_6_Free-Solid-900.otf", 14, fa_loading_params)

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

env_map = gfx.TextureMap(env_tex)

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
inspector_textures = []


def collect_inspector_targets(obj):
    state["current_inspector_target"] = None
    inspector_materials.clear()
    inspector_textures.clear()

    def visit(obj):
        if (
            hasattr(obj, "material")
            and obj.material is not None
            and obj.material not in inspector_materials
        ):
            if not obj.material.name:
                obj.material.name = f"{type(obj.material).__name__}"

            inspector_materials.append(obj.material)

            for name, value in inspect.getmembers(obj.material):
                if isinstance(value, gfx.TextureMap):
                    if value not in inspector_textures:
                        if not value.name:
                            value.name = f"{obj.material.name} ({name})"

                        inspector_textures.append(value)

    obj.traverse(visit)


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
            model_obj.traverse(lambda obj: add_env_map(obj, env_map))

        skeleton_helper = gfx.SkeletonHelper(model_obj)
        skeleton_helper.visible = False
        scene.add(skeleton_helper)
        scene.add(model_obj)
        state["selected_action"] = 0

        camera.show_object(model_obj, scale=1.4)

        if actions:
            for action in actions:
                action.stop()
            actions = []

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
                    model_obj.traverse(lambda obj: add_env_map(obj, env_map))
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
    imgui.set_next_window_size((800, 800), imgui.Cond_.appearing)

    p_open, state["show_inspector"] = imgui.begin("Inspector", state["show_inspector"])

    if p_open:
        if imgui.begin_child(
            "##tree", (300, 0), imgui.ChildFlags_.resize_x | imgui.ChildFlags_.borders
        ):
            if model_obj:
                node_open = imgui.tree_node_ex(
                    "Nodes",
                    imgui.TreeNodeFlags_.span_full_width
                    | imgui.TreeNodeFlags_.allow_overlap,
                )
                if node_open:

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
                            icon = icons_fontawesome_6.ICON_FA_LIGHTBULB
                            icon_color = imgui.IM_COL32(255, 215, 0, 255)
                        elif isinstance(obj, gfx.Camera):
                            icon = icons_fontawesome_6.ICON_FA_CAMERA
                            icon_color = imgui.IM_COL32(0, 128, 0, 255)
                        elif isinstance(obj, gfx.Bone):
                            icon = icons_fontawesome_6.ICON_FA_BONE
                            icon_color = imgui.IM_COL32(255, 255, 255, 255)

                        imgui.push_style_color(imgui.Col_.text, icon_color)
                        imgui.same_line(spacing=1)
                        imgui.text(icon)
                        imgui.pop_style_color()

                        imgui.same_line(spacing=5)
                        imgui.text(obj.name or "<Unnamed>")

                        if isinstance(obj, gfx.Mesh):
                            imgui.same_line()

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
                                imgui.push_style_color(
                                    imgui.Col_.text, (0.5, 0.5, 0.5, 1)
                                )

                            imgui.set_cursor_pos_x(
                                imgui.get_cursor_pos_x()
                                + imgui.get_content_region_avail().x
                                - 48
                            )

                            if imgui.small_button(
                                icons_fontawesome_6.ICON_FA_EYE
                                if obj.visible
                                else icons_fontawesome_6.ICON_FA_EYE_SLASH
                            ):
                                obj.visible = not obj.visible

                            imgui.pop_style_color()

                            if imgui.is_item_hovered():
                                imgui.set_tooltip("Toggle visibility")

                            imgui.same_line()
                            imgui.set_cursor_pos_x(
                                imgui.get_cursor_pos_x()
                                + imgui.get_content_region_avail().x
                                - 24
                            )

                            if (
                                getattr(obj, "_box_helper", None) is not None
                                and obj._box_helper.visible
                            ):
                                imgui.push_style_color(imgui.Col_.text, (1, 1, 1, 1))
                            else:
                                imgui.push_style_color(
                                    imgui.Col_.text, (0.5, 0.5, 0.5, 1)
                                )

                            if imgui.small_button(icons_fontawesome_6.ICON_FA_SQUARE):
                                if getattr(obj, "_box_helper", None) is None:
                                    box_helper = gfx.BoxHelper()

                                    obj._box_helper = box_helper
                                    scene.add(box_helper)
                                    obj._box_helper.visible = False

                                obj._box_helper.visible = not obj._box_helper.visible
                                if obj._box_helper.visible:
                                    obj._box_helper.set_transform_by_object(obj)

                            if imgui.is_item_hovered():
                                imgui.set_tooltip("Toggle bounding box")

                            imgui.pop_style_color()

                            imgui.pop_style_color(3)

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
                                f"{icons_fontawesome_6.ICON_FA_BRUSH}",
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

                if imgui.tree_node("Textures"):
                    if inspector_textures:
                        for tex in inspector_textures:
                            imgui.push_id(str(id(tex)))
                            imgui.push_style_color(
                                imgui.Col_.text, imgui.IM_COL32(147, 112, 219, 255)
                            )
                            _, clicked = imgui.selectable(
                                f"{icons_fontawesome_6.ICON_FA_IMAGE}",
                                state["current_inspector_target"] == tex,
                                imgui.SelectableFlags_.span_all_columns,
                            )
                            imgui.pop_style_color()
                            if clicked:
                                state["current_inspector_target"] = tex

                            imgui.same_line(spacing=10)
                            imgui.text(tex.name)
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
            elif isinstance(target, gfx.TextureMap):
                show_texture_map_properties(target)

        imgui.end_group()
    imgui.end()


def show_material_properties(material):
    imgui.separator_text("Transparency Settings")

    alpha_config = material.alpha_config
    alpha_mode = alpha_config["mode"]

    alpha_modes = [
        "solid",
        "solid_premul",
        "blend",
        "add",
        "subtract",
        "dither",
        "bayer",
        "weighted_blend",
        "weighted_solid",
    ]

    imgui.set_next_item_width(200)
    current_idx = alpha_modes.index(alpha_mode) if alpha_mode in alpha_modes else 0
    changed, alpha_mode_idx = imgui.combo("Alpha Mode", current_idx, alpha_modes)

    if changed:
        alpha_mode = alpha_modes[alpha_mode_idx]
        material.alpha_mode = alpha_mode

    imgui.text(f"Render Queue: {material.render_queue}")

    imgui.begin_table(
        "Alpha Config", 2, imgui.TableFlags_.row_bg | imgui.TableFlags_.resizable
    )

    for key, value in alpha_config.items():
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text(key)
        imgui.table_next_column()
        imgui.text(f"{value}")

    imgui.end_table()

    imgui.set_next_item_width(200)
    _, material.opacity = imgui.slider_float(
        "Opacity",
        material.opacity,
        0.0,
        1.0,
    )

    imgui.set_next_item_width(200)
    _, material.alpha_test = imgui.slider_float(
        "Alpha Test", material.alpha_test, 0.0, 1.0
    )

    _, material.depth_write = imgui.checkbox("Depth Write", material.depth_write)
    _, material.depth_test = imgui.checkbox("Depth Test", material.depth_test)

    imgui.separator_text("Detailed Properties")

    imgui.begin_table(
        "Details", 2, imgui.TableFlags_.row_bg | imgui.TableFlags_.resizable
    )
    for name, value in inspect.getmembers(material):
        if not name.startswith("_") and not callable(value):
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text(name)
            imgui.table_next_column()
            if isinstance(value, (int, float, str)):
                imgui.text(str(value))
            elif isinstance(value, (list, tuple)):
                imgui.text(", ".join(f"{v:.2f}" for v in value))
            elif isinstance(value, gfx.TextureMap):
                imgui.text_colored(
                    (147 / 255, 112 / 255, 219 / 255, 1),
                    icons_fontawesome_6.ICON_FA_IMAGE,
                )
                imgui.same_line(spacing=5)
                if imgui.text_link(f"{value.name}##{name}"):
                    state["current_inspector_target"] = value

            elif isinstance(value, gfx.Color):
                imgui.text(
                    f"Color: {value.r:.2f}, {value.g:.2f}, {value.b:.2f}, {value.a:.2f}"
                )
            else:
                imgui.text(str(value))

    imgui.end_table()


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
    imgui.text(f"X: {position[0]:.2f}, Y: {position[1]:.2f}, Z: {position[2]:.2f}")
    imgui.table_next_row()
    imgui.table_next_column()
    imgui.text("Quaternion")
    imgui.table_next_column()
    imgui.text(
        f"X: {rotation[0]:.2f}, Y: {rotation[1]:.2f}, Z: {rotation[2]:.2f}, W: {rotation[3]:.2f}"
    )
    imgui.table_next_row()
    imgui.table_next_column()
    imgui.text("Scale")
    imgui.table_next_column()
    imgui.text(f"X: {scale[0]:.2f}, Y: {scale[1]:.2f}, Z: {scale[2]:.2f}")
    imgui.end_table()

    if isinstance(obj, gfx.Mesh):
        imgui.separator_text("Mesh Info")
        imgui.begin_table("Mesh", 2)
        vertices = obj.geometry.positions.nitems
        faces = obj.geometry.indices.nitems
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text("Vertices")
        imgui.table_next_column()
        imgui.text(str(vertices))
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text("Faces")
        imgui.table_next_column()
        imgui.text(str(faces))

        if obj.material:
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Active material")
            imgui.table_next_column()
            imgui.text_colored(
                (255 / 255, 165 / 255, 0 / 255, 1), icons_fontawesome_6.ICON_FA_BRUSH
            )
            imgui.same_line(spacing=5)
            if imgui.text_link(f"{obj.material.name}"):
                state["current_inspector_target"] = obj.material

        imgui.end_table()


def show_texture_map_properties(tex_map):
    if getattr(tex_map, "_imgui_tex", None) is None:
        if getattr(tex_map.texture, "_wgpu_object", None) is not None:
            tex_map._imgui_tex = gui_renderer.backend.register_texture(
                tex_map.texture._wgpu_object.create_view()
            )

    if getattr(tex_map, "_imgui_tex", None):
        img_pos_x = max(0, (imgui.get_content_region_avail().x - 300) / 2)
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + img_pos_x)
        # Display the texture map image, keeping the aspect ratio
        size = tex_map.texture.size
        aspect_ratio = size[1] / size[0] if size[0] > 0 else 1.0
        imgui.image(tex_map._imgui_tex, (300, 300 * aspect_ratio))

    imgui.separator_text("Properties")

    imgui.begin_table(
        "TextureMap", 2, imgui.TableFlags_.row_bg | imgui.TableFlags_.resizable
    )
    for k, v in inspect.getmembers(tex_map):
        if not k.startswith("_") and not callable(v):
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text(k)
            imgui.table_next_column()
            imgui.text(str(v))

    imgui.end_table()

    imgui.separator()
    imgui.begin_table(
        "Texture", 2, imgui.TableFlags_.row_bg | imgui.TableFlags_.resizable
    )

    if tex_map.texture is not None:
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text("Source")
        imgui.table_next_column()
        imgui.text(f"{tex_map.texture.name or 'no name'}")
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text("dimensions")
        imgui.table_next_column()
        imgui.text(f"{tex_map.texture.dim}D, {tex_map.texture.size}")
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text("format")
        imgui.table_next_column()
        imgui.text(tex_map.texture.format)
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text("colorspace")
        imgui.table_next_column()
        imgui.text(tex_map.texture.colorspace)
    imgui.end_table()


gui_renderer.set_gui(draw_imgui)
# load_remote_model(0)

load_model(assets_dir / "Soldier.glb")


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
