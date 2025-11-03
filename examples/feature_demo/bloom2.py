"""
Bloom Effect Demo
=================

This example shows a bloom rendering effect.
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
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx

from wgpu.utils.imgui import ImguiRenderer, Stats
from imgui_bundle import imgui

# Init
canvas = RenderCanvas(
    size=(1280, 720), title="bloom_effect", update_mode="fastest", vsync=False
)
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

# Apply env map to skybox
background = gfx.Background.from_color((0.0, 0.0, 0.0, 1))
scene.add(background)

# Load meshes, and apply env map
# Note that this lights the helmet already
gltf_path = model_dir / "phoenix_bird.glb"

gltf = gfx.load_gltf(gltf_path)
# gfx.print_scene_graph(gltf.scene) # Uncomment to see the tree structure


def add_env_map(obj, env_map):
    if isinstance(obj, gfx.Mesh) and isinstance(obj.material, gfx.MeshStandardMaterial):
        obj.material.env_map = env_map


gltf.scene.traverse(lambda obj: add_env_map(obj, env_tex))

scene.add(gltf.scene)

# Add extra light more or less where the sun seems to be in the skybox
light = gfx.SpotLight(color="#444")
light.local.position = (-500, 1000, -1000)
scene.add(light)

# Create bloom effect pass using the new API
bloom_pass = gfx.renderers.wgpu.PhysicalBasedBloomPass(
    bloom_strength=0.4,
    max_mip_levels=6,
    filter_radius=0.005,
    use_karis_average=False,
)

# Add bloom pass to renderer's effect passes
renderer.effect_passes = [bloom_pass]

action_clip = gltf.animations[0]

animation_mixer = gfx.AnimationMixer()
action = animation_mixer.clip_action(action_clip)
action.play()

# Create camera and controller
camera = gfx.PerspectiveCamera(45, 16 / 9)
camera.show_object(gltf.scene)
controller = gfx.OrbitController(camera, register_events=renderer)

clock = gfx.Clock()

gui_renderer = ImguiRenderer(renderer.device, canvas)


def draw_imgui():
    imgui.set_next_window_size((400, 0), imgui.Cond_.always)
    imgui.set_next_window_pos((0, 0), imgui.Cond_.always)
    imgui.begin("Bloom Settings")

    changed, value = imgui.checkbox("bloom", bloom_pass.enable)
    if changed:
        bloom_pass.enable = value

    changed, value = imgui.slider_float(
        "Bloom Strength", bloom_pass.bloom_strength, 0.0, 3.0
    )
    if changed:
        bloom_pass.bloom_strength = value

    changed, value = imgui.slider_int(
        "Max Mipmap Levels", bloom_pass.max_mip_levels, 1, 10
    )
    if changed:
        bloom_pass.max_mip_levels = value

    changed, value = imgui.slider_float(
        "Filter Radius", bloom_pass.filter_radius, 0.0, 0.01
    )
    if changed:
        bloom_pass.filter_radius = value

    changed, value = imgui.checkbox("Use Karis Average", bloom_pass.use_karis_average)
    if changed:
        bloom_pass.use_karis_average = value

    imgui.end()


gui_renderer.set_gui(draw_imgui)
stats = Stats(renderer.device, canvas, align="right")


def animate():
    dt = clock.get_delta()

    animation_mixer.update(dt)
    with stats:
        renderer.render(scene, camera)
        gui_renderer.render()
    canvas.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    loop.run()
