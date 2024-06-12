"""
Camera linking 2
================

Example demonstating the partial linking of two cameras in 3D. The left
and right view use an orbit- and fly-controller, respectively. Yet,
manipulating one also updates the othe camera.

This demonstrates how the controllers are stateless (except during interaction).
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas(size=(900, 600)))


# Create a scene to display multiple times. It contains a point cloud
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#ddd"))

n = 1000
positions = np.random.normal(0, 50, (n, 3)).astype(np.float32)
sizes = np.random.rand(n).astype(np.float32) * 50
colors = np.random.rand(n, 4).astype(np.float32)

points = gfx.Points(
    gfx.Geometry(positions=positions, sizes=sizes, colors=colors),
    gfx.PointsMaterial(color_mode="vertex", size_mode="vertex", size_space="world"),
)
scene.add(points)


# Background
viewport0 = gfx.Viewport(renderer)
camera0 = gfx.NDCCamera()
scene0 = gfx.Background.from_color("#fff")

# Create view 1
viewport1 = gfx.Viewport(renderer)
camera1 = gfx.PerspectiveCamera(50)
camera1.show_object(scene)
controller1 = gfx.OrbitController(camera1, register_events=viewport1)

# Create view 2
viewport2 = gfx.Viewport(renderer)
camera2 = gfx.PerspectiveCamera(50)
camera2.show_object(scene)
controller2 = gfx.FlyController(camera2, register_events=viewport2)

# Link cameras
controller1.add_camera(camera2, include_state={"position", "rotation"})
controller2.add_camera(camera1, include_state={"position", "rotation"})


@renderer.add_event_handler("resize")
def layout(event=None):
    w, h = renderer.logical_size
    w2 = (w - 30) / 2
    viewport1.rect = 10, 10, w2, h - 20
    viewport2.rect = w2 + 20, 10, w2, h - 20


# Initialize layout
layout()


def animate():
    viewport0.render(scene0, camera0)
    viewport1.render(scene, camera1)
    viewport2.render(scene, camera2)
    renderer.flush()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
