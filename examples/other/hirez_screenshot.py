"""
High resolution screenshot
==========================

"""

# sphinx_gallery_pygfx_docs = 'code'
# sphinx_gallery_pygfx_test = 'off'

import os
import tempfile
import webbrowser

import imageio as iio
import numpy as np
from rendercanvas.offscreen import RenderCanvas
import pygfx as gfx


# %% Prepare

# The scale factor determines how many tiles are created to build the
# final screenshot (upscale_factor**2). Using a higher value allows
# creating screenshots above the wgpu texture size limit.
upscale_factor = 4

# Determine the canvas size, and thereby the resolution of each tile.
# The tile_size and full_size that we calculate below are both in logical pixels.
canvas_size = 1200, 1000

# Maintain logical size or make it upscale_factor times larger
maintain_logical_size = True

if maintain_logical_size:
    # The logical size is maintained, so the result is an image with a
    # pixel_ratio of `upscale_factor`. Things sized in screen coordinates
    # (like the texts on the left) scale the same as the rest.
    tile_size = canvas_size[0] // upscale_factor, canvas_size[1] // upscale_factor
    full_size = tile_size[0] * upscale_factor, tile_size[1] * upscale_factor
else:
    # The logical size is made upscale_factor as large. The pixel_ratio
    # is 1. Things sized in screen coordinates (like the texts on the
    # left) become tiny compared to the rest.
    tile_size = canvas_size
    full_size = tile_size[0] * upscale_factor, tile_size[1] * upscale_factor

canvas = RenderCanvas(size=canvas_size, pixel_ratio=1)
renderer = gfx.WgpuRenderer(canvas)


# %% The visualization

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
scene.add(gfx.Background.from_color("#bbb", "#777", "#f00", "#0f0"))

y = 0
text = gfx.Text(
    text="centered",
    anchor="middle-center",
    font_size=1,
    material=gfx.TextMaterial("#000"),
)
text.local.y = y
text.local.x = npoints
scene.add(text)

text = gfx.Text(
    text="inner",
    anchor="middle-center",
    font_size=1,
    material=gfx.TextMaterial("#000"),
)
text.local.y = y
text.local.x = 2 * npoints + npoints
scene.add(text)

text = gfx.Text(
    text="outer",
    anchor="middle-center",
    font_size=1,
    material=gfx.TextMaterial("#000"),
)
text.local.y = y
text.local.x = 4 * npoints + npoints
scene.add(text)

all_points = []
for marker in gfx.MarkerShape:
    if marker == "custom":
        continue
    y += 2
    points = gfx.Points(
        geometry,
        gfx.PointsMarkerMaterial(
            size=1,
            size_space="world",
            color_mode="vertex",
            marker=marker,
            edge_color="#000",
            edge_width=0.1,
        ),
    )
    points.local.y = -y
    points.local.x = 1
    scene.add(points)
    all_points.append(points)

    points_inner = gfx.Points(
        geometry,
        gfx.PointsMarkerMaterial(
            size=1,
            size_space="world",
            color_mode="vertex",
            marker=marker,
            edge_color="#000",
            edge_width=0.1,
            edge_mode="inner",
        ),
    )

    points_inner.local.y = -y
    points_inner.local.x = 1 + 2 * npoints

    scene.add(points_inner)
    all_points.append(points_inner)

    points_outer = gfx.Points(
        geometry,
        gfx.PointsMarkerMaterial(
            size=1,
            size_space="world",
            color_mode="vertex",
            marker=marker,
            edge_color="#000",
            edge_width=0.1,
            edge_mode="outer",
        ),
    )

    points_outer.local.y = -y
    points_outer.local.x = 1 + 4 * npoints

    scene.add(points_outer)
    all_points.append(points_outer)

    text = gfx.Text(
        text=marker,
        anchor="middle-right",
        font_size=20,
        screen_space=True,
        material=gfx.TextMaterial("#000"),
    )
    text.local.y = -y
    text.local.x = 0
    scene.add(text)


camera = gfx.OrthographicCamera()
camera.show_object(scene, scale=0.8)

camera_state = camera.get_state()

# To help visualize the rendering ability, we draw two boxes
# One entirely contained within the FOV of the main image, the other
# entirely outside of the FOV of the main image.
# By disabling AA, we should be able to use this to validate that we don't
# render more than we should.
line_thickness = 0.1
box_width = camera_state["width"]
box_height = camera_state["height"]
logical_size = canvas.get_logical_size()
if logical_size[0] > logical_size[1]:
    box_width *= logical_size[0] / logical_size[1]
else:
    box_height *= logical_size[1] / logical_size[0]
inner_lines_geometry = gfx.box_geometry(
    width=box_width - line_thickness,
    height=box_height - line_thickness,
)


inner_lines_geometry.positions.data[..., 0] += camera_state["position"][0]
inner_lines_geometry.positions.data[..., 1] += camera_state["position"][1]

outer_lines_geometry = gfx.Geometry(
    positions=[
        [0, 0, 0],
        [box_width + line_thickness, 0, 0],
        [box_width + line_thickness, box_height + line_thickness, 0],
        [0, box_height + line_thickness, 0],
        [0, 0, 0],
    ]
)
outer_lines_geometry.positions.data[..., 0] -= (box_width + line_thickness) / 2
outer_lines_geometry.positions.data[..., 1] -= (box_height + line_thickness) / 2

outer_lines_geometry.positions.data[..., 0] += camera_state["position"][0]
outer_lines_geometry.positions.data[..., 1] += camera_state["position"][1]
scene.add(
    gfx.Line(
        inner_lines_geometry,
        gfx.LineMaterial(
            color="blue", thickness=line_thickness, thickness_space="world", aa=False
        ),
    )
)
scene.add(
    gfx.Line(
        outer_lines_geometry,
        gfx.LineMaterial(
            color="red", thickness=line_thickness, thickness_space="world", aa=False
        ),
    )
)


## Tiling


@canvas.request_draw
def animate():
    renderer.render(scene, camera)


# Create snapshot of tiles.
# A possible improvement would be to write each tile-row once it is captured,
# so we never need the full image as one contiguous array. This would enable
# creating massive screenshots even on machines with little RAM.
rows = []
for iy in range(upscale_factor):
    row = []
    for ix in range(upscale_factor):
        camera.set_view_offset(
            full_size[0],
            full_size[1],
            ix * tile_size[0],
            iy * tile_size[1],
            tile_size[0],
            tile_size[1],
        )
        im = np.asarray(canvas.draw())
        # im = im[:,:,:3]  # rgba -> rgb
        row.append([im])  # the list-nesting is to make block work correctly
    rows.append(row)

# Safe full image
full_im = np.block(rows)
print("full resolution:", full_im.shape)
filename = os.path.join(tempfile.gettempdir(), "hirez_pygfx.png")
iio.imwrite(filename, full_im)
print(f"{os.stat(filename).st_size / 2**20:0.3f} MiB")

# Show the image
webbrowser.open("file://" + filename)
