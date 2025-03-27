"""
Line loop
=========

Drawing a line with a loop.
"""

# ruff: noqa: RUF005

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx


canvas = RenderCanvas(size=(1000, 800))
renderer = gfx.WgpuRenderer(canvas)
scene = gfx.Scene()

rect_points = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], "f4")
nanpoint = np.full((1, 3), np.nan, dtype="f4")


# rect_points = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0.8, 0, 0], [0.8, 0.8, 0], [0.2, 0.8, 0], [0.2, 0, 0], [0.5, 0, 0], [0.5, 0.6, 0]],"f4")

positions1 = rect_points * 10

positions2 = np.vstack(
    [
        rect_points,
        nanpoint,
        rect_points + (2, 0, 0),
        nanpoint,
        rect_points + (0, 2, 0),
        nanpoint,
    ],
    dtype="f4",
)

line1 = gfx.Line(
    gfx.Geometry(positions=positions2),
    gfx.LineMaterial(thickness=20, color="red", opacity=0.7, loop=1),
)


scene.add(line1)

camera = gfx.OrthographicCamera(600, 500)
camera.show_object(scene, match_aspect=True, scale=1.1)
controller = gfx.PanZoomController(camera, register_events=renderer)


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    loop.run()
