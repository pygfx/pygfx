"""
Stencil outline (object selection)
==================================

A common use of the stencil buffer is drawing an outline around the
selected object. Here the selected object first writes its silhouette
into the stencil buffer (``compare="always"``, ``pass_op="replace"``).
A slightly larger copy of the object is then drawn with
``compare="not_equal"``, so it only appears where the stencil buffer does
*not* contain the silhouette - i.e. as a rim around the object.

Click an object to select it.

Also see the multi_select.py example.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import math
import time

import wgpu
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx
import pylinalg as la


canvas = RenderCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas, enable_stencil=True)
scene = gfx.Scene()

scene.add(gfx.Background.from_color("#1d2129"))

# The stencil reference value used to mark the silhouette of the selected object
STENCIL_MARK = 1

objects = []
selected = None


def make_selectable(geometry, material, position):
    ob = gfx.Mesh(geometry, material)
    ob.local.position = position
    scene.add(ob)
    objects.append(ob)

    # The outline: a slightly larger copy of the object, drawn after the
    # object itself (higher render_queue), but only where the object did
    # not write its silhouette into the stencil buffer.
    outline = gfx.Mesh(
        geometry,
        gfx.MeshBasicMaterial(color="#ffa500", depth_write=False, render_queue=3000),
    )
    outline.local.scale = (1.06, 1.06, 1.06)
    outline.visible = False
    outline.material.stencil_value = STENCIL_MARK
    outline.material.stencil_front_state = wgpu.StencilFaceState(compare="not-equal")
    ob.add(outline)
    ob.outline = outline
    return ob


torus = make_selectable(
    gfx.torus_knot_geometry(60, 12, 128, 32),
    gfx.MeshPhongMaterial(color="#4cc9f0", pick_write=True),
    (-110, 0, 0),
)
cube = make_selectable(
    gfx.box_geometry(120, 120, 120),
    gfx.MeshPhongMaterial(color="#80ed99", pick_write=True),
    (110, 0, 0),
)
sphere = make_selectable(
    gfx.sphere_geometry(70),
    gfx.MeshPhongMaterial(color="#f72585", pick_write=True),
    (0, 0, -160),
)


def select(ob):
    global selected
    if ob is selected:
        return
    if selected is not None:
        # Stop writing the stencil buffer, hide the outline
        selected.material.stencil_front_state = None
        selected.outline.visible = False
    selected = ob
    # Mark the object's silhouette in the stencil buffer ...
    ob.material.stencil_value = STENCIL_MARK
    ob.material.stencil_front_state = wgpu.StencilFaceState(
        compare="always", pass_op="replace"
    )
    # ... and show its outline
    ob.outline.visible = True


def on_pointer_down(event):
    if event.target in objects:
        select(event.target)


for ob in objects:
    ob.add_event_handler(on_pointer_down, "pointer_down")

select(torus)  # initial selection

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.show_object(scene, view_dir=(0, 0, -1), up=(0, 1, 0))
controller = gfx.OrbitController(camera, register_events=renderer)

scene.add(gfx.AmbientLight(0.3), camera.add(gfx.DirectionalLight(1.0)))


def animate():
    rot = la.quat_from_euler((0.003, 0.007), order="XY")
    for ob in objects:
        ob.local.rotation = la.quat_mul(rot, ob.local.rotation)

    # Make the outline pulse a bit
    if selected is not None:
        s = 1.06 + 0.015 * math.sin(time.perf_counter() * 3)
        selected.outline.local.scale = (s, s, s)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    print(__doc__)
    canvas.request_draw(animate)
    loop.run()
