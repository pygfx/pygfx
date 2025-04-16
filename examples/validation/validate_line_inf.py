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
circle_positions = np.column_stack([x, y, np.zeros_like(x)])

# More points
more_positions = np.array(
    [
        [-1, 0, 1],  # Two points in the same position stay a single point
        [-1, 0, 1],
        [1, -1, 1],  # A vertical line
        [1, 1, 1],
    ],
    np.float32,
)

# Combine
positions = np.vstack([circle_positions, more_positions])

# Assign color to the different line segments
colors = np.zeros_like(positions)
colors[: len(circle_positions)] = (0, 0, 0.8)
colors[-4:-2] = (1, 1, 1)
colors[-2:] = (1, 1, 0)


line = gfx.Line(
    gfx.Geometry(positions=positions, colors=colors),
    gfx.LineInfSegmentMaterial(thickness=10.0, color_mode="face", map=gfx.cm.viridis),
)
scene.add(line)

camera = gfx.OrthographicCamera(maintain_aspect=True)
camera.show_object(scene, match_aspect=True, scale=2)
controller = gfx.PanZoomController(camera, register_events=renderer)

canvas.request_draw(lambda: renderer.render(scene, camera))


if __name__ == "__main__":
    print(__doc__)
    loop.run()
