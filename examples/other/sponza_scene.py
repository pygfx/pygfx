"""
Sponza Scene
============

This example shows how to load the Sponza scene.
"""
# run_example = false - because it depends on external files

from pathlib import Path

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


# run the following command in the root of this repo
# git clone https://github.com/KhronosGroup/glTF-Sample-Models ../glTF-Sample-Models
# so that the paths will be set up correctly
repo_root = Path(__file__).parent.parent
gltf_samples_repo = repo_root.parent / "glTF-Sample-Models"
gltf_path = gltf_samples_repo / "2.0" / "Sponza" / "glTF" / "Sponza.gltf"

# Init
canvas = WgpuCanvas(size=(640, 480), title="gltf viewer")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# Create camera and controller
camera = gfx.PerspectiveCamera(45, 640 / 480)
camera.position.set(-10, 1.8, -0.3)
# look down the hall, which is positive x
target = camera.position.clone().add(gfx.linalg.Vector3(10))
controller = gfx.OrbitController(camera.position.clone(), target)
controller.add_default_event_handlers(renderer, camera)


def configure(obj):
    obj.receive_shadow = True
    obj.cast_shadow = True


# Load scene
meshes = gfx.load_scene(gltf_path)
scene.add(*meshes)
scene.traverse(configure)

# Add ambient light
scene.add(gfx.AmbientLight(intensity=0.1))

# Add the sun, midday direction
sunlight = gfx.DirectionalLight()
sunlight.position.set(-14.5, 31, 4.5)
sunlight.target.position.set(5.3, -1.4, -2.5)
sunlight.cast_shadow = True
scene.add(sunlight)

# Add torches
for pos in [
    [-5.0, 1.09, -1.75],
    [-5.0, 1.09, 1.15],
    [3.8, 1.09, -1.75],
    [3.8, 1.09, 1.15],
]:
    torch = gfx.PointLight("#ff7700", decay=2.5)
    torch.position.set(*pos)
    torch.cast_shadow = True
    scene.add(torch)


def animate():
    controller.update_camera(camera)
    renderer.render(scene, camera)


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
