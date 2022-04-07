"""
Like scene_side_by_side, but now with a more plot-like idea, and mouse interaction.
"""

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())


# Create a scene to display multiple times. It contains three points clouds.
scene = gfx.Scene()
scene.add(gfx.Background(None, gfx.BackgroundMaterial("#ddd")))

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
camera0 = gfx.NDCCamera()
scene0 = gfx.Background(None, gfx.BackgroundMaterial("#fff"))

# Create view 1 - xy
camera1 = gfx.OrthographicCamera(8, 8)
controls1 = gfx.PanZoomControls(
    gfx.linalg.Vector3(0, 0, 1),
    gfx.linalg.Vector3(0, 0, 0),
    gfx.linalg.Vector3(0, 1, 0),
)
controls1.add_default_event_handlers(renderer, camera1)

# Create view 2 - xz
camera2 = gfx.OrthographicCamera(8, 8)
controls2 = gfx.PanZoomControls(
    gfx.linalg.Vector3(0, 1, 0),
    gfx.linalg.Vector3(0, 0, 0),
    gfx.linalg.Vector3(0, 0, 1),
)
controls2.add_default_event_handlers(renderer, camera2)

# Create view 3 - yz
camera3 = gfx.OrthographicCamera(8, 8)
controls3 = gfx.PanZoomControls(
    gfx.linalg.Vector3(1, 0, 0),
    gfx.linalg.Vector3(0, 0, 0),
    gfx.linalg.Vector3(0, 0, 1),
)
controls3.add_default_event_handlers(renderer, camera3)

# Create view 4 - 3D
camera4 = gfx.OrthographicCamera(8, 8)
controls4 = gfx.OrbitControls(
    gfx.linalg.Vector3(1, 1, 1),
    gfx.linalg.Vector3(0, 0, 0),
    gfx.linalg.Vector3(0, 0, 1),
    zoom_changes_distance=False,
)
controls4.add_default_event_handlers(renderer, camera4)


@renderer.add_event_handler("resize")
def layout(event=None):
    w, h = renderer.logical_size
    w2, h2 = (w - 30) / 2, (h - 30) / 2
    controls1.viewport = 10, 10, w2, h2
    controls2.viewport = w / 2 + 5, 10, w2, h2
    controls3.viewport = 10, h / 2 + 5, w2, h2
    controls4.viewport = w / 2 + 5, h / 2 + 5, w2, h2


layout()


def animate():

    controls1.update_camera(camera1)
    controls2.update_camera(camera2)
    controls3.update_camera(camera3)
    controls4.update_camera(camera4)

    renderer.render(scene0, camera0, flush=False)
    renderer.render(scene, camera1, viewport=controls1.viewport, flush=False)
    renderer.render(scene, camera2, viewport=controls2.viewport, flush=False)
    renderer.render(scene, camera3, viewport=controls3.viewport, flush=False)
    renderer.render(scene, camera4, viewport=controls4.viewport, flush=False)
    renderer.flush()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
