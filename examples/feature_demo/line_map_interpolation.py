"""
Line map interpolation
=======================

The texture coordinates that index a line's colormap are interpolated across
the line. This is what you want for a continuous colormap, but for a categorical
one it makes a segment that spans two categories sample the categories in
between. Setting ``LineMaterial.map_interpolation = "flat"`` disables the
interpolation, so each part of the line shows a single color.

The top line uses 'flat' interpolation, the bottom line the default 'perspective'.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import numpy as np
import cmap
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx

canvas = RenderCanvas()
renderer = gfx.WgpuRenderer(canvas)

# A line that represents categorical data, e.g. cluster labels
x = np.linspace(0, 4 * np.pi, 10)
y = np.sin(x)
positions = np.column_stack([x, y, np.zeros(10)]).astype(np.float32)

labels = np.repeat([0, 2, 4, 6, 8], 2).astype(np.float32)
# Place each label at the center of its texel in the 10-color map.
texcoords = ((labels + 0.5)).astype(np.float32)

geometry = gfx.Geometry(positions=positions, texcoords=texcoords)

tab10 = cmap.Colormap("tab10").to_pygfx()

line_flat = gfx.Line(
    geometry,
    gfx.LineMaterial(
        thickness=20,
        color_mode="vertex_map",
        map=tab10,
        map_interpolation="flat",
        maprange=(0, 10),  # the range of tab10 is [0, 10)
    ),
)
line_perspective = gfx.Line(
    geometry,
    gfx.LineMaterial(
        thickness=20, color_mode="vertex_map", map=tab10, maprange=(0, 10)
    ),
)
line_perspective.local.y = -2

scene = gfx.Scene()
scene.add(line_flat, line_perspective)

camera = gfx.OrthographicCamera()
camera.show_object(scene)
controller = gfx.OrbitController(camera, register_events=renderer)

canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    loop.run()
