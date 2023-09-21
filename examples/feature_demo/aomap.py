"""
Ambient occlusion
=================


This example demonstrates the ambient occlusion map effects for MeshBasicMaterial, MeshPhongMaterial, and MeshStandardMaterial.
"""

################################################################################
# .. warning::
#     An external model is needed to run this example.
#
# To run this example, you need a model from the source repo's example
# folder. If you are running this example from a local copy of the code (dev
# install) no further actions are needed. Otherwise, you may have to replace
# the path below to point to the location of the model.

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

# sphinx_gallery_pygfx_render = True

import imageio.v3 as iio
import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


# Init
canvas = WgpuCanvas(size=(1200, 400), title="aomap")
renderer = gfx.renderers.WgpuRenderer(canvas)

meshes = gfx.load_meshes(model_dir / "lightmap" / "scene.gltf")

ao_map = iio.imread(model_dir / "lightmap" / "lightmap-ao-shadow.png")
ao_map_tex = gfx.Texture(ao_map, dim=2)

texcoords1 = np.ascontiguousarray(
    np.loadtxt(model_dir / "lightmap" / "texcoords1.txt"), dtype="f4"
)
texcoords1 = gfx.Buffer(texcoords1)

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
    m.geometry.texcoords1 = texcoords1
    material.ao_map = ao_map_tex
    mesh = gfx.Mesh(m.geometry, material)
    scene.add(mesh)

    # illumination the scene for MeshPhongMaterial and MeshStandardMaterial
    scene.add(gfx.AmbientLight(intensity=1.0))

    t = gfx.Text(
        gfx.TextGeometry(material.__class__.__name__, screen_space=True, font_size=20),
        gfx.TextMaterial(),
    )
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
