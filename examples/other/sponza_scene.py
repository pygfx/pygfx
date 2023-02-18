"""
Sponza Scene
============

This example shows how to load the Sponza scene. To run it, you have to have access to the
sponza demo scene, which you can get using::

    git clone https://github.com/KhronosGroup/glTF-Sample-Models ../glTF-Sample-Models

The current implementation assumed that you do this in the root directory of pygfx,
so if you don't you will need to update the path in the script below.

"""
# run_example = false - because it depends on external files

from pathlib import Path

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


# Note: __file__ is the relative path from cwd when called as "__main__". Hence,
# we need to resolve() to allow calling from anywhere in the repo.
repo_root = Path(__file__).resolve().parents[2]

# !! Update this path to point to the location of the sponza scene file !!
# If it is located in pygfx's root, you don't need to do anything
gltf_samples_repo = repo_root / "glTF-Sample-Models"
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
meshes = gfx.load_scene(gltf_path.as_posix())
scene.add(*meshes)
scene.traverse(configure)

# Add ambient light
scene.add(gfx.AmbientLight(intensity=0.1))

# Add the sun, midday direction
sunlight = gfx.DirectionalLight()
sunlight.position.set(-14.5, 31, 4.5)
sunlight.target.position.set(5.3, -1.4, -2.5)
sunlight.cast_shadow = True
sunlight.shadow.camera.near = 0
sunlight.shadow.camera.far = 50
sunlight.shadow.camera.update_projection_matrix()
scene.add(sunlight)

# Add torches
for pos in [
    [-5.0, 1.12, -1.75],
    [-5.0, 1.12, 1.15],
    [3.8, 1.12, -1.75],
    [3.8, 1.12, 1.15],
]:
    torch = gfx.PointLight("#ff7700", decay=2.5)
    torch.position.set(*pos)
    torch.cast_shadow = True
    torch.shadow.camera.near = 0.01
    torch.shadow.camera.update_projection_matrix()
    # torch.add(gfx.PointLightHelper(size=0.01))
    scene.add(torch)


def animate():
    controller.update_camera(camera)
    renderer.render(scene, camera)


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
