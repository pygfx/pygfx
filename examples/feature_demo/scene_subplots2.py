"""
Subplots 2
==========

Like scene_side_by_side, but now with a more plot-like idea, and mouse interaction.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())


# Create a scene to display multiple times. It contains three points clouds.
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#ddd"))

for clr, offset in [
    ("red", (-3, -1, 0)),
    ("green", (0, 0, 0)),
    ("blue", (3, 1, 0)),
]:
    N = 100
    positions = np.random.normal(size=(N, 3)) + np.array(offset)
    colors = np.repeat([clr], N, 0)

    points = gfx.Points(
        gfx.Geometry(positions=positions.astype(np.float32)),
        gfx.PointsMaterial(size=10, color=clr),
    )
    scene.add(points)


# Background
viewport0 = gfx.Viewport(renderer)
camera0 = gfx.NDCCamera()
scene0 = gfx.Background.from_color("#fff")

# Create view 1 - xy
viewport1 = gfx.Viewport(renderer)
camera1 = gfx.OrthographicCamera(8, 8)
camera1.show_rect(-3, 3, -3, 3, view_dir=(0, 0, -1), up=(0, 1, 0))
controller1 = gfx.PanZoomController(camera1, register_events=viewport1)

# Create view 2 - xz
viewport2 = gfx.Viewport(renderer)
camera2 = gfx.OrthographicCamera(8, 8)
camera2.show_rect(-3, 3, -3, 3, view_dir=(0, -1, 0), up=(0, 0, 1))
controller2 = gfx.PanZoomController(camera2, register_events=viewport2)

# Create view 3 - yz
viewport3 = gfx.Viewport(renderer)
camera3 = gfx.OrthographicCamera(8, 8)
camera3.show_rect(-3, 3, -3, 3, view_dir=(-1, 0, 0), up=(0, 0, 1))
controller3 = gfx.PanZoomController(camera3, register_events=viewport3)

# Create view 4 - 3D
viewport4 = gfx.Viewport(renderer)
camera4 = gfx.OrthographicCamera(8, 8)
camera4.show_rect(-3, 3, -3, 3, view_dir=(-1, -1, -1), up=(0, 0, 1))
controller4 = gfx.OrbitController(camera4, register_events=viewport4)


@renderer.add_event_handler("resize")
def layout(event=None):
    w, h = renderer.logical_size
    w2, h2 = (w - 30) / 2, (h - 30) / 2
    viewport1.rect = 10, 10, w2, h2
    viewport2.rect = w / 2 + 5, 10, w2, h2
    viewport3.rect = 10, h / 2 + 5, w2, h2
    viewport4.rect = w / 2 + 5, h / 2 + 5, w2, h2


# Initialize layout
layout()


def animate():
    viewport0.render(scene0, camera0)
    viewport1.render(scene, camera1)
    viewport2.render(scene, camera2)
    viewport3.render(scene, camera3)
    viewport4.render(scene, camera4)
    renderer.flush()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
