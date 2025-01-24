"""
Sponza Scene
============

This example shows how to load the Sponza scene. To run it, you have
to have access to the sponza demo scene, which you can get using::

    git clone https://github.com/KhronosGroup/glTF-Sample-Assets

The current implementation assumes that you cloned that repo
in this directory, or *any* of its parent directories.

"""

# sphinx_gallery_pygfx_docs = 'code'
# sphinx_gallery_pygfx_test = 'off'

from pathlib import Path

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


# Find assets directory
for path in Path(__file__).resolve().parents:
    gltf_samples_dir = path / "glTF-Sample-Assets"
    if gltf_samples_dir.is_dir():
        break
else:
    raise RuntimeError("Could not find 'glTF-Sample-Assets' directory.")

gltf_path = gltf_samples_dir / "Models" / "Sponza" / "glTF" / "Sponza.gltf"

# Init
canvas = WgpuCanvas(size=(640, 480), title="gltf viewer")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# Create camera and controller
camera = gfx.PerspectiveCamera(45, 640 / 480)
camera.local.position = -10, 1.8, -0.3
camera.show_pos((0, 0, 0))
controller = gfx.FlyController(camera, register_events=renderer, speed=2)


def configure(obj):
    obj.receive_shadow = True
    obj.cast_shadow = True


# Load scene
scene.add(*gfx.load_mesh(gltf_path.as_posix()))
scene.traverse(configure)

# Add ambient light
scene.add(gfx.AmbientLight(intensity=0.1))

# Add the sun, midday direction
sunlight = gfx.DirectionalLight()
sunlight.local.position = -14.5, 31, 4.5
sunlight.target.local.position = 5.3, -1.4, -2.5
sunlight.cast_shadow = True
sunlight.shadow.camera.depth_range = (0, 250)
scene.add(sunlight)

# Add torches
for pos in [
    [-5.0, 1.12, -1.75],
    [-5.0, 1.12, 1.15],
    [3.8, 1.12, -1.75],
    [3.8, 1.12, 1.15],
]:
    torch = gfx.PointLight("#ff7700", decay=2.5)
    torch.local.position = pos
    torch.cast_shadow = True
    torch.shadow.camera.depth_range = (0.01, 200)
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
