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

controls1 = gfx.PanZoomControls(camera1.position.clone())
controls1.add_default_event_handlers(renderer, camera1)
controls1.look_at(
    gfx.linalg.Vector3(0, 0, 0),
    gfx.linalg.Vector3(0, 0, 1),
    gfx.linalg.Vector3(0, 1, 0),
)

# Create view 2 - xz
camera2 = gfx.OrthographicCamera(8, 8)

controls2 = gfx.PanZoomControls(camera2.position.clone())
controls2.add_default_event_handlers(renderer, camera2)
controls1.look_at(
    gfx.linalg.Vector3(0, 0, 0),
    gfx.linalg.Vector3(0, 1, 0),
    gfx.linalg.Vector3(0, 0, 1),
)

# Create view 3 - yz
camera3 = gfx.OrthographicCamera(8, 8)

controls3 = gfx.PanZoomControls(camera3.position.clone())
controls3.add_default_event_handlers(renderer, camera3)
controls3.look_at(
    gfx.linalg.Vector3(0, 0, 0),
    gfx.linalg.Vector3(1, 0, 0),
    gfx.linalg.Vector3(0, 0, 1),
)


def animate():

    w, h = renderer.logical_size
    w2, h2 = (w - 30) / 2, (h - 30) / 2
    viewport1 = 10, 10, w2, h2
    viewport2 = w / 2 + 5, 10, w2, h2
    viewport3 = 10, h / 2 + 5, w2, h2

    controls1.update_camera(camera1)
    controls2.update_camera(camera2)
    controls3.update_camera(camera3)

    controls1.update_viewport(viewport1)
    controls2.update_viewport(viewport2)
    controls3.update_viewport(viewport3)

    renderer.render(scene0, camera0, flush=False)
    renderer.render(scene, camera1, viewport=viewport1, flush=False)
    renderer.render(scene, camera2, viewport=viewport2, flush=False)
    renderer.render(scene, camera3, viewport=viewport3, flush=False)
    renderer.flush()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
