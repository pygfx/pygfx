"""
This example shows how to load a gltf file.
"""

from pathlib import Path

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
from pygfx.utils.gltf import load_gltf


# check out this repo: https://github.com/KhronosGroup/glTF-Sample-Models
# and update the paths to load one of the models in there
repo_root = Path(__file__).parent.parent
gltf_samples_repo = repo_root.parent / "glTF-Sample-Models"
gltf_path = gltf_samples_repo / "2.0" / "Sponza" / "glTF" / "Sponza.gltf"

# Init
canvas = WgpuCanvas(size=(640, 480), title="gltf viewer")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# Create camera and controller
camera = gfx.PerspectiveCamera(45, 640 / 480)
camera.position.set(-1.8, 0.6, 2.7)
controller = gfx.OrbitController(camera.position.clone())
controller.add_default_event_handlers(renderer, camera)

# Load meshes, and apply env map
meshes = load_gltf(gltf_path)
scene.add(*meshes)

# Ensure there is _some_ light
scene.add(gfx.AmbientLight())

scene.add(gfx.PointLight(position=(0, 5, 0), distance=5))
scene.add(gfx.PointLight(position=(0, 2, 3), distance=5, color="green"))


def animate():
    controller.update_camera(camera)
    renderer.render(scene, camera)


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
