"""
Infinite line segments
======================

Display infinite line segments.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import numpy as np
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx


canvas = RenderCanvas()
renderer = gfx.WgpuRenderer(canvas)

scene = gfx.Scene()

# Create an even number of points in a circle
t = np.linspace(0, 2 * np.pi, 22, endpoint=False, dtype=np.float32)
x = np.sin(t) * 4
y = np.cos(t) * 4
positions1 = np.column_stack([x, y, np.zeros_like(x)])

# More points
positions2 = np.array(
    [
        [-1, 0, 1],  # Two points in the same position stay a single point
        [-1, 0, 1],
        [1, -1, 1],  # A vertical line
        [1, 1, 1],
    ],
    np.float32,
)


# Assign color to the different line segments
colors1 = np.zeros_like(positions1)
colors1[:, 0] = 1
colors1[:, 1] = np.linspace(0, 1, len(colors1))


line1 = gfx.Line(
    gfx.Geometry(positions=positions1, colors=colors1),
    gfx.LineInfSegmentMaterial(
        thickness=5.0, color_mode="face", start_is_infinite=False
    ),
)
scene.add(line1)

line2 = gfx.Line(
    gfx.Geometry(positions=positions2),
    gfx.LineInfSegmentMaterial(thickness=10.0, color="cyan"),
)
scene.add(line2)

camera = gfx.OrthographicCamera(maintain_aspect=True)
camera.show_object(scene, match_aspect=True, scale=2)
controller = gfx.PanZoomController(camera, register_events=renderer)

canvas.request_draw(lambda: renderer.render(scene, camera))


if __name__ == "__main__":
    print(__doc__)
    loop.run()
