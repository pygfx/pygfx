"""
Earth
=====


This example demonstrates the effect of rendering a globe using MeshPhongMaterial.
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
import pylinalg as la
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


# Init
canvas = WgpuCanvas(size=(640, 480), title="Earth")
renderer = gfx.renderers.WgpuRenderer(canvas)

scene = gfx.Scene()

earth_geometry = gfx.sphere_geometry(63.71, 100, 50)


def load_texture(path, flip=False):
    img = iio.imread(path)
    if flip:
        img = np.ascontiguousarray(np.flipud(img))
    tex = gfx.Texture(img, dim=2)
    return tex


# Earth
earth_material = gfx.MeshPhongMaterial(specular="#333333", shininess=20)
earth_material.map = load_texture(
    model_dir / "planets" / "earth_atmos_4096.jpg", flip=True
)
earth_material.specular_map = load_texture(
    model_dir / "planets" / "earth_specular_2048.jpg", flip=True
)
earth_material.emissive = "#444433"
earth_material.emissive_map = load_texture(
    model_dir / "planets" / "earth_lights_2048.png", flip=True
)
earth_material.emissive_intensity = 2.5

# # Uncomment to use light map instead of emissive map
# earth_material.light_map = earth_material.emissive_map
# earth_geometry.texcoords1 = earth_geometry.texcoords
# earth_material.light_map_intensity = 2.5

earth_material.normal_map = load_texture(
    model_dir / "planets" / "earth_normal_2048.jpg", flip=True
)
earth_material.normal_scale = 0.85, -0.85

earth = gfx.Mesh(earth_geometry, earth_material)
scene.add(earth)

# Clouds
clouds_material = gfx.MeshPhongMaterial()
clouds_map_tex = load_texture(
    model_dir / "planets" / "earth_clouds_2048.png", flip=True
)
clouds_material.map = clouds_map_tex
clouds_material.opacity = 0.9
clouds_material.side = "Front"

earth_clouds = gfx.Mesh(earth_geometry, clouds_material)
earth_clouds.local.scale = 1.005, 1.005, 1.005  # Slightly larger than earth
earth_clouds.local.rotation = la.quat_from_euler((0, 0.0, 0.41), order="XYZ")
scene.add(earth_clouds)

# Create camera and controller
camera = gfx.PerspectiveCamera(45, 640 / 480, depth_range=(1, 10000))
camera.local.position = 0, 0, 300
camera.look_at((0, 0, 0))
controller = gfx.OrbitController(camera, register_events=renderer)

# Add lights
sun_light = gfx.DirectionalLight(color=(1, 1, 1), intensity=1.5)
sun_light.local.position = 3, 0, 1
scene.add(sun_light)
scene.add(gfx.AmbientLight(intensity=0.05))

rot = la.quat_from_euler((0, 0.001, 0), order="XYZ")
rot_cloud = la.quat_from_euler((0, 0.00125, 0), order="XYZ")


def animate():
    earth.local.rotation = la.quat_mul(rot, earth.local.rotation)
    earth_clouds.local.rotation = la.quat_mul(rot_cloud, earth_clouds.local.rotation)
    renderer.render(scene, camera)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
