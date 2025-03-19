"""
Lightmap
========

This example demonstrates the lightmap effects for MeshBasicMaterial, MeshPhongMaterial, and MeshStandardMaterial.
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


# Init
canvas = WgpuCanvas(size=(1200, 400), title="lightmap")
renderer = gfx.renderers.WgpuRenderer(canvas)

meshes = gfx.load_gltf_mesh(model_dir / "lightmap" / "scene.gltf", materials=False)

light_map = iio.imread(model_dir / "lightmap" / "lightmap-ao-shadow.png")
light_map_tex = gfx.Texture(light_map, dim=2)

# Create camera and controller
camera = gfx.PerspectiveCamera(45, 1)
camera.local.position = 500, 1000, 1500
camera.look_at((0, 0, 0))
controller = gfx.OrbitController(camera, register_events=renderer)

# Text
text_scene = gfx.Scene()
text_camera = gfx.OrthographicCamera(12, 4)


def create_scene(material, x_pos):
    scene = gfx.Scene()
    m = meshes[0]
    material.light_map = gfx.TextureMap(light_map_tex, uv_channel=1)
    mesh = gfx.Mesh(m.geometry, material)
    mesh.local.scale = 100
    scene.add(mesh)

    t = gfx.Text(text=material.__class__.__name__, screen_space=True, font_size=20)
    t.local.position = (x_pos, 1.5, 0)
    text_scene.add(t)

    return scene


vp1 = gfx.Viewport(renderer, (5, 0, 390, 400))
scene1 = create_scene(gfx.MeshBasicMaterial(), -4)

vp2 = gfx.Viewport(renderer, (405, 0, 390, 400))
scene2 = create_scene(gfx.MeshPhongMaterial(), 0)

vp3 = gfx.Viewport(renderer, (805, 0, 390, 400))
scene3 = create_scene(gfx.MeshStandardMaterial(), 4)

vp4 = gfx.Viewport(renderer, (0, 0, 1200, 400))


def animate():
    vp1.render(scene1, camera)
    vp2.render(scene2, camera)
    vp3.render(scene3, camera)
    vp4.render(text_scene, text_camera)
    renderer.flush()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
