"""
Points with different markers
=============================

* All available marker shapes are shown.
* Shows red, green and blue faces. Then a semi-transparent face, and finally a fully-transparent face.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas(size=(600, 1000))
renderer = gfx.WgpuRenderer(canvas)

colors = np.array(
    [
        [1.0, 0.5, 0.5, 1.0],
        [0.5, 1.0, 0.5, 1.0],
        [0.5, 0.5, 1.0, 1.0],
        [0.5, 0.5, 1.0, 0.3],
        [0.0, 0.0, 0.0, 0.0],
    ],
    np.float32,
)

npoints = len(colors)

positions = np.zeros((npoints, 3), np.float32)
positions[:, 0] = np.arange(npoints) * 2
geometry = gfx.Geometry(positions=positions, colors=colors)


scene = gfx.Scene()
scene.add(gfx.Background(None, gfx.BackgroundMaterial("#bbb", "#777")))

y = 0
for marker in gfx.MarkerShape:
    y += 2
    line = gfx.Points(
        geometry,
        gfx.PointsMarkerMaterial(
            size=30,
            color_mode="vertex",
            marker=marker,
            edge_color="#000",
            edge_width=3,
        ),
    )
    line.local.y = -y
    line.local.x = 1
    scene.add(line)

    text = gfx.Text(
        gfx.TextGeometry(
            marker, anchor="middle-right", font_size=20, screen_space=True
        ),
        gfx.TextMaterial("#000"),
    )
    text.local.y = -y
    text.local.x = 0
    scene.add(text)

camera = gfx.OrthographicCamera()
camera.show_object(scene, scale=0.7)

canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    run()
