"""
Line join seam
==============

* A row of increasingly sharp corners, dashed on top and solid below.
* The dash phase is chosen so that a dash straddles each corner.
* The lines are translucent, which is what makes the defect plain.

**This example currently renders incorrectly**, on purpose: it is the picture of
an open bug. Every pixel of a uniformly translucent line over a flat background
must composite to the same value, and on the dashed row the inside of each
corner from 70 degrees down is crossed by a darker hairline.

A line overlaps itself wherever a join is "broken" -- a corner too sharp to be
mitred, which the shader covers with the two segments' own caps instead. Those
faces are coplanar and disagree about coverage where they overlap, so the depth
test arbitrates, and the fragment that wins may be the one carrying a partial
antialiasing alpha. That is the seam.

Dashing is what makes these corners broken joins at all: the shader mitres up to
`max_vec_mag`, which is 100 when solid but drops to 1.5 (about 90 degrees) when
dashing. Hence the solid row below is clean at the same angles -- it is the
control, not a second example of the bug.

The corners are drawn separately rather than as one polygon so that each gets
the same dash phase, which is what makes the row comparable.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx


canvas = RenderCanvas(size=(1000, 500))
renderer = gfx.WgpuRenderer(canvas)
renderer.ppaa = "none"  # the seam is the shader's own, not the AA pass's

scene = gfx.Scene()
scene.add(gfx.Background.from_color("#000"))

THICKNESS = 14.0
# The dash phase at the corner is leg_length / THICKNESS, in dash units, so this
# puts the corner one unit into a two-unit stroke: a dash sits astride it.
LEG_LENGTH = 70.0
ANGLES = [90, 70, 60, 45, 30]


def corner(angle_deg, cx, cy):
    """Two legs meeting at `angle_deg`, opening upwards from (cx, cy)."""
    half = np.radians(angle_deg / 2)
    dx, dy = LEG_LENGTH * np.sin(half), LEG_LENGTH * np.cos(half)
    return np.array(
        [[cx - dx, cy + dy, 0], [cx, cy, 0], [cx + dx, cy + dy, 0]], np.float32
    )


for i, angle in enumerate(ANGLES):
    x = -380 + i * 190
    for dashed, y, color in (
        (True, 40, (1.0, 0.85, 0.33, 0.4)),
        (False, -190, (0.47, 0.68, 1.0, 0.4)),
    ):
        scene.add(
            gfx.Line(
                gfx.Geometry(positions=corner(angle, x, y)),
                gfx.LineMaterial(
                    thickness=THICKNESS,
                    color=color,
                    aa=True,
                    dash_pattern=[2, 2] if dashed else (),
                ),
            )
        )
    label = gfx.Text(
        text=f"{angle}",
        font_size=18,
        screen_space=True,
        anchor="middle-center",
        material=gfx.TextMaterial(color="#888"),
    )
    label.local.position = (x, -60, 0)
    scene.add(label)

camera = gfx.OrthographicCamera(1000, 500)
controller = gfx.PanZoomController(camera, register_events=renderer)

canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    loop.run()
