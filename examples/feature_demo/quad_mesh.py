"""
Mesh with quads
===============

This example demonstrates the use of meshes with quad faces.
Quad faces are used when geometry.indices array is Nx4 instead of Nx3.

This example also demonstrates per-vertex and per-face coloring.

Contributed by S. Shaji
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


def generate_sample_quads(cols=9):
    pos = np.dstack(np.meshgrid(np.arange(cols), np.arange(2))).reshape(-1, 2)
    z = np.abs([*np.arange(-cols / 2, cols / 2), *np.arange(-cols / 2, cols / 2)])
    pos = np.c_[pos, z].astype("f")
    n1 = np.arange(cols)
    n2 = np.full(cols - 1, 0)
    n3 = np.full(cols - 1, 1)
    idx = np.dstack((n1[:-1], n2, n1[:-1], n3, n1[1:], n3, n1[1:], n2)).reshape(-1, 2)
    indices = (idx[:, 0] + idx[:, 1] * cols).reshape(-1, 4)
    return pos, indices.astype(np.int32)


canvas = WgpuCanvas(
    title="Mesh Object with quads. Press 1,2 or 3 for wireframe, per vertex coloring or per face coloring"
)
renderer = gfx.renderers.WgpuRenderer(canvas)

# Show something
scene = gfx.Scene()
camera = gfx.PerspectiveCamera()

controller = gfx.OrbitController(camera=camera, register_events=renderer)
controller.controls["mouse3"] = ("pan", "drag", (1.0, 1.0))

# Generate Sample quads and draw them
pos, indices = generate_sample_quads()
colors = np.repeat(pos[:, -1] / pos[:, -1].max(), 4).reshape(-1, 4)
colors[:, 2] = 1
colors[:, 3] = 1

patches = gfx.Mesh(
    gfx.Geometry(
        indices=indices,
        positions=pos,
        colors=colors,
        texcoords=np.linspace(0, 1, len(indices), dtype=np.float32),
    ),
    gfx.MeshBasicMaterial(
        color_mode="uniform",
        wireframe=True,
        wireframe_thickness=3,
        map=gfx.cm.magma,
        pick_write=True,
    ),
)


scene.add(patches)
camera.show_object(patches)

# Let there be ...
scene.add(gfx.AmbientLight())
light = gfx.DirectionalLight()
light.local.position = (0, 0, 1)

# Create a contrasting background
clr = [i / 255 for i in [87, 188, 200, 255]]
background = gfx.Background.from_color(clr)
scene.add(background)


def make_wireframe():
    patches.material.wireframe = True
    patches.material.color_mode = "uniform"


def make_vertex_color():
    patches.material.wireframe = False
    patches.material.color_mode = "vertex"


def make_face_color():
    patches.material.wireframe = False
    patches.material.color_mode = "face"


def make_face_color_map():
    patches.material.wireframe = False
    patches.material.color_mode = "face_map"


@renderer.add_event_handler("key_down")
def on_key(e):
    if e.key == "1":
        make_wireframe()
    elif e.key == "2":
        make_vertex_color()
    elif e.key == "3":
        make_face_color()
    elif e.key == "4":
        make_face_color_map()


@renderer.add_event_handler("click")
def pick_id(event):
    print(event.pick_info)


def animate():
    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
