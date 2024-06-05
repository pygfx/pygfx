"""
Subplots 3
==========

bla
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())


# Create a scene to display multiple times. It contains a point cloud
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#ddd"))

n = 5000
x = np.linspace(0, 100, n)
y = np.random.normal(size=(n, 1))
z = np.zeros_like(x)

points = gfx.Points(
    gfx.Geometry(positions=np.column_stack([x, y, z]).astype(np.float32)),
    gfx.PointsMaterial(size=10, color="#066", opacity=0.5),
)
scene.add(points)


# Background
viewport0 = gfx.Viewport(renderer)
camera0 = gfx.NDCCamera()
scene0 = gfx.Background.from_color("#fff")

# Create view 1
viewport1 = gfx.Viewport(renderer)
camera1 = gfx.OrthographicCamera(8, 8, maintain_aspect=False)
camera1.show_rect(0, 30, -4, 4, view_dir=(0, 0, -1), up=(0, 1, 0))
controller1 = gfx.PanZoomController(camera1, register_events=viewport1, )

# Create view 2 - xz
viewport2 = gfx.Viewport(renderer)
camera2 = gfx.OrthographicCamera(8, 8, maintain_aspect=False)
camera2.show_rect(0, 30, -2, 2, view_dir=(0, 0, -1), up=(0, 1, 0))
controller2 = gfx.PanZoomController(camera2, register_events=viewport2)

controller1.add_camera(camera2, {"position_x", "width"})
controller2.add_camera(camera1, {"position_x", "width"})


controller1.controls
@renderer.add_event_handler("resize")
def layout(event=None):
    w, h = renderer.logical_size
    w2, h2 = (w - 30) / 2, (h - 30) / 2
    viewport1.rect = 10, 10, w - 20, h2
    viewport2.rect = 10, h2 + 20, w - 20, h2


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
