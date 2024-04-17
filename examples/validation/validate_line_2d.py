"""
Lines in 2D
===========

* This shows a 2D line in 3 flavours.
* It shows a gap due to nan values.
* It has duplicate points to test correct handling for that.
* It also puts a few points in quick succession to test that it staus a continuous line (with joins, no caps).
* It includes a point that only has its z-value different, which should behave like a duplicate point.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas(size=(1000, 1000))
renderer = gfx.WgpuRenderer(canvas)
renderer.blend_mode = "weighted"

positions = [
    # Cap
    [-1, 0, 0],
    # Some joins / corners
    [0, 1, 0],
    [1, -0.5, 0],
    [2, 0, 0],
    # nan and inf create gaps, should trigger when *any* field is nonfinite.
    # 4 gaps in total.
    [2.5, 0, 1],
    [np.nan, np.nan, np.nan],
    [3, 0, 1],
    [np.nan, 0, 1],
    [3.5, 0, 1],
    [3.75, np.inf, 1],
    [4.0, 0, 1],
    [4.25, 0, -np.inf],
    [4.5, 0, 1],
    [5, 0, 1],
    # A duplicate point
    [6, 1, 1],
    [7, 1, 1],
    [7, 1, 1],
    [8, 1, 1],
    # A triplicate point
    [9, 2, 1],
    [10, 2, 1],
    [10, 2, 1],
    [10, 2, 1],
    [11, 2, 1],
    # A bunch of points close together
    [12.0, 3, 1],
    [12.3, 3, 1],
    [12.4, 3, 1],
    [12.5, 3, 1],
    [12.6, 3, 1],
    [12.7, 3, 1],
    [13.0, 3, 1],
    # A seemingly duplicate point, but different in z
    [14, 4, 1],
    [15, 4, 1],
    [15, 4, 0],
    [16, 4, 0],
]
colors = [
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 1, 1],
    [0, 0, 1],
    [1, 0, 1],
]

positions = np.array(positions, np.float32)
colors = np.tile(np.array(colors, np.float32), (len(positions) // 6 + 1, 1))
colors = colors[: len(positions)]

print(colors.shape)

geometry = gfx.Geometry(positions=positions, colors=colors)

line1 = gfx.Line(
    geometry,
    gfx.LineDebugMaterial(
        thickness=40,
        thickness_space="screen",
        color=(0.0, 1.0, 1.0),
        opacity=1.0,
    ),
)

line2 = gfx.Line(
    geometry,
    gfx.LineMaterial(
        thickness=40,
        thickness_space="screen",
        color_mode="vertex",
        opacity=0.5,
    ),
)

line3 = gfx.Line(
    geometry,
    gfx.LineMaterial(
        thickness=20,
        thickness_space="screen",
        color_mode="uniform",
        dash_pattern=[1, 1.2],
        color=(1.0, 0.0, 0.0),
        opacity=0.5,
    ),
)


line1.local.position = 0, 4, 0
line3.local.position = 0, -4, 0

scene = gfx.Scene()
scene.add(line1, line2, line3)


camera = gfx.OrthographicCamera()
camera.show_object(scene)

controller = gfx.OrbitController(camera, register_events=renderer)


def animate():
    renderer.render(scene, camera)
    for ob in scene.iter():
        if isinstance(ob, gfx.Line):
            if hasattr(ob.material, "dash_offset"):
                ob.material.dash_offset += 0.1


canvas.request_draw(animate)

if __name__ == "__main__":
    print(__doc__)
    run()
