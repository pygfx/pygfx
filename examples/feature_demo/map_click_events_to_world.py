"""
Map Screen to World
===================

This shows how to map from the screen to the world
by adding scatter points at click event locations
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from wgpu.gui.auto import WgpuCanvas, run

import pygfx as gfx
import numpy as np

from pylinalg import vec_transform, vec_unproject

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)

# default cam position
center_cam_pos = (256, 256, 0)


# lists of everything necessary to make this plot
scenes = list()
cameras = list()
controllers = list()
camera_defaults = list()
viewports = list()

bg_colors = ["r", "g", "b", "y"]

for i in range(4):
    # create scene for this subplot
    scene = gfx.Scene()
    scenes.append(scene)

    bg = gfx.Background.from_color(bg_colors[i])
    scene.add(bg)

    # create camera, set default position, add to list
    camera = gfx.OrthographicCamera(512, 512)
    camera.show_rect(0, 512, 0, 512)
    cameras.append(camera)

    # create viewport
    viewport = gfx.Viewport(renderer)
    viewports.append(viewport)

    # controller for pan & zoom
    controller = gfx.PanZoomController(camera, register_events=renderer)
    controllers.append(controller)

    # get the initial controller params so the camera can be reset later
    camera_defaults.append(camera.get_state())


@renderer.add_event_handler("resize")
def layout(event=None):
    """
    Update the viewports when the canvas is resized
    """
    w, h = renderer.logical_size
    w2, h2 = w / 2 - 15, h / 2 - 15
    viewports[0].rect = 10, 10, w2, h2
    viewports[1].rect = w / 2 + 5, 10, w2, h2
    viewports[2].rect = 10, h / 2 + 5, w2, h2
    viewports[3].rect = w / 2 + 5, h / 2 + 5, w2, h2


def animate():
    # render the viewports
    for viewport, s, c in zip(viewports, scenes, cameras):
        viewport.render(s, c)

    renderer.flush()
    canvas.request_draw()


def add_point(ev):
    """Add point at mouse click location by mapping click position in canvas/screen space to world space"""

    # this is the click position in canvas space
    pos = (ev.x, ev.y)
    print(f"position click: {pos}")

    for viewport, scene, camera in zip(viewports, scenes, cameras):
        if not viewport.is_inside(*pos):
            continue

        # get position relative to viewport
        pos_rel = (
            pos[0] - viewport.rect[0],
            pos[1] - viewport.rect[1],
        )

        vs = viewport.logical_size

        # convert position to NDC
        x = pos_rel[0] / vs[0] * 2 - 1
        y = -(pos_rel[1] / vs[1] * 2 - 1)
        pos_ndc = (x, y, 0)

        pos_ndc += vec_transform(camera.world.position, camera.camera_matrix)
        pos_world = vec_unproject(pos_ndc[:2], camera.camera_matrix)

        # make point data
        point_data = np.array([[pos_world[0], pos_world[1], 0]], dtype=np.float32)

        # add point
        point = gfx.Points(
            gfx.Geometry(positions=point_data),
            material=gfx.PointsMaterial(color="black", size=10),
        )

        scene.add(point)


renderer.add_event_handler(add_point, "click")
layout()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
