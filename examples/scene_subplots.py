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
view0 = gfx.View(renderer, camera=gfx.NDCCamera())
view0.scene.add(gfx.Background(None, gfx.BackgroundMaterial("#fff")))

# Create view 1 - xy
view1 = gfx.View(renderer, scene=scene)
view1.camera = gfx.OrthographicCamera(8, 8)

controls1 = gfx.PanZoomControls(view1.camera.position.clone())
controls1.add_default_event_handlers(view1)
controls1.look_at(
    gfx.linalg.Vector3(0, 0, 0),
    gfx.linalg.Vector3(0, 0, 1),
    gfx.linalg.Vector3(0, 1, 0),
)

# Create view 2 - xz
view2 = gfx.View(renderer, scene=scene)
view2.camera = gfx.OrthographicCamera(8, 8)

controls2 = gfx.PanZoomControls(view2.camera.position.clone())
controls2.add_default_event_handlers(view2)
controls1.look_at(
    gfx.linalg.Vector3(0, 0, 0),
    gfx.linalg.Vector3(0, 1, 0),
    gfx.linalg.Vector3(0, 0, 1),
)

# Create view 3 - yz
view3 = gfx.View(renderer, scene=scene)
view3.camera = gfx.OrthographicCamera(8, 8)

controls3 = gfx.PanZoomControls(view3.camera.position.clone())
controls3.add_default_event_handlers(view3)
controls3.look_at(
    gfx.linalg.Vector3(0, 0, 0),
    gfx.linalg.Vector3(1, 0, 0),
    gfx.linalg.Vector3(0, 0, 1),
)


def animate():

    controls1.update_camera(view1.camera)
    controls2.update_camera(view2.camera)
    controls3.update_camera(view3.camera)

    w, h = renderer.logical_size
    w2, h2 = (w - 30) / 2, (h - 30) / 2
    view1.viewport = 10, 10, w2, h2
    view2.viewport = w / 2 + 5, 10, w2, h2
    view3.viewport = 10, h / 2 + 5, w2, h2

    renderer.render_views(view0, view1, view2, view3)


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
