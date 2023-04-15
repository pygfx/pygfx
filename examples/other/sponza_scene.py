"""
Sponza Scene
============

This example shows how to load the Sponza scene. To run it, you have
to have access to the sponza demo scene, which you can get using::

    git clone https://github.com/KhronosGroup/glTF-Sample-Models

The current implementation assumes that you cloned that repo
in this directory, or *any* of its parent directories.

"""
# run_example = false - because it depends on external files

from pathlib import Path

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


# Note: __file__ is the relative path from cwd when called as "__main__". Hence,
# we need to resolve() to allow calling from anywhere in the repo.
# Note: if we want a better way to find that repo, let's use an env var ...
path = Path(__file__).resolve().parent
while True:
    gltf_samples_repo = path / "glTF-Sample-Models"
    if gltf_samples_repo.is_dir():
        break
    try:
        path = path.parents[0]
    except IndexError:
        raise RuntimeError("Could not find 'glTF-Sample-Models' directory.")


gltf_path = gltf_samples_repo / "2.0" / "Sponza" / "glTF" / "Sponza.gltf"

# Init
canvas = WgpuCanvas(size=(640, 480), title="gltf viewer")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# Create camera and controller
camera = gfx.PerspectiveCamera(45, 640 / 480)
camera.position.set(-10, 1.8, -0.3)
camera.show_pos((0, 0, 0))
controller = gfx.FlyController(camera, register_events=renderer, speed=2)


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
sunlight.shadow.camera.depth_range = (0, 250)
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
    torch.shadow.camera.depth_range = (0.01, 200)
    torch.shadow.camera.update_projection_matrix()
    torch.shadow.cull_mode = "none"
    torch.shadow.bias = 0.001
    # torch.add(gfx.PointLightHelper(size=0.01))
    scene.add(torch)


stats = gfx.Stats(viewport=renderer)


def animate():
    with stats:
        renderer.render(scene, camera, flush=False)
    stats.render()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
