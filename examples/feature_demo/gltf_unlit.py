"""
glTF Unlit Material Example
===========================

This example demonstrates unlit materials in glTF models.
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

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
from wgpu.utils.imgui import ImguiRenderer
from imgui_bundle import imgui

# Init
canvas = WgpuCanvas(size=(1280, 720), title="gltf_unlit")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# Read cube image and turn it into a 3D image (a 4d array)
env_img = iio.imread("imageio:meadow_cube.jpg")
cube_size = env_img.shape[1]
env_img.shape = 6, cube_size, cube_size, env_img.shape[-1]

# Create environment map
env_tex = gfx.Texture(
    env_img, dim=2, size=(cube_size, cube_size, 6), generate_mipmaps=True
)

# Load meshes, and apply env map
# Note that this lights the helmet already
gltf_path = model_dir / "just_a_girl" / "scene.gltf"

gltf = gfx.load_gltf(gltf_path)
# gfx.print_scene_graph(gltf.scene)  # Uncomment to see the tree structure


def add_env_map(obj, env_map):
    if isinstance(obj, gfx.Mesh) and isinstance(obj.material, gfx.MeshStandardMaterial):
        obj.material.env_map = env_map


unlit_model = gltf.scene if gltf.scene else gltf.scenes[0]
unlit_model.local.position = (-100, 0, 0)
scene.add(unlit_model)


from pygfx.utils.load_gltf import _GLTF  # noqa

# Hack to remove the unlit plugin
parser = _GLTF()
del parser._plugins["KHR_materials_unlit"]

pbr_gltf = parser.load(gltf_path, quiet=True)
pbr_model = pbr_gltf.scene if pbr_gltf.scene else pbr_gltf.scenes[0]
pbr_model.traverse(lambda obj: add_env_map(obj, env_tex))
pbr_model.local.position = (100, 0, 0)
scene.add(pbr_model)

gui_renderer = ImguiRenderer(renderer.device, canvas)


state = {"ibl": True}


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
        _, direct_light.visible = imgui.checkbox(
            "Directional Light", direct_light.visible
        )

        _, ambient_light.visible = imgui.checkbox(
            "Ambient Light", ambient_light.visible
        )

        changed, state["ibl"] = imgui.checkbox("IBL", state["ibl"])
        if changed:
            if state["ibl"]:
                pbr_model.traverse(lambda obj: add_env_map(obj, env_tex))
            else:
                pbr_model.traverse(lambda obj: add_env_map(obj, None))

    imgui.end()
    imgui.end_frame()
    imgui.render()
    return imgui.get_draw_data()


gui_renderer.set_gui(draw_imgui)


# Add extra light more or less where the sun seems to be in the skybox
direct_light = gfx.DirectionalLight(intensity=2.5)
direct_light.local.position = (-5, 10, -10)
scene.add(direct_light)

ambient_light = gfx.AmbientLight(intensity=0.2)
scene.add(ambient_light)

# Create camera and controller
camera = gfx.PerspectiveCamera(45, 640 / 480)
camera.show_object(scene)
controller = gfx.OrbitController(camera, register_events=renderer)


def animate():
    renderer.render(scene, camera)
    gui_renderer.render()
    canvas.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
