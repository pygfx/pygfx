"""
Clipping Plane Caps
===================

Example closing the openings that clipping planes cut into a closed mesh, using
the stencil buffer (similar to three.js ``webgl_clipping_stencil``).

Per plane, the mesh clipped by only that plane is drawn with
``color_write=False``, flipping a stencil bit for each of its surfaces. Where
the bit ends up set (an odd number of surfaces), the plane cuts through the
inside of the mesh, and the cap is drawn there.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import wgpu
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx
import pylinalg as la


canvas = RenderCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas, enable_stencil=True)
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#dcdcdc"))

planes = [(-1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0)]  # keep x, y and z <= 0
cap_colors = ["#a8d5ba", "#e8a5b8", "#e8c39e"]

geometry = gfx.torus_knot_geometry(1, 0.3, 128, 32)
knot = gfx.Mesh(
    geometry,
    gfx.MeshPhongMaterial(color="#5588ff", clipping_planes=planes),
)
scene.add(knot)

invert = wgpu.StencilFaceState(compare="always", pass_op="invert")
not_zero = wgpu.StencilFaceState(compare="not-equal")  # stencil_value is 0

cap_geometry = gfx.plane_geometry(4, 4)
for i, (plane, color) in enumerate(zip(planes, cap_colors, strict=True)):
    bit = 1 << i  # each plane uses its own stencil bit

    # Flip the bit for each surface
    count = gfx.Mesh(
        geometry,
        gfx.MeshBasicMaterial(
            color_write=False,
            depth_test=False,
            depth_write=False,
            side="both",
            clipping_planes=[plane],
            stencil_front_state=invert,
            stencil_back_state=invert,
            stencil_masks=(bit, bit),
        ),
    )
    count.render_order = 1
    knot.add(count)  # turns with the knot

    # Draw the cap where the bit is set
    cap = gfx.Mesh(
        cap_geometry,
        gfx.MeshPhongMaterial(
            color=color,
            clipping_planes=[p for p in planes if p != plane],
            stencil_front_state=not_zero,
            stencil_back_state=not_zero,
            stencil_masks=(bit, bit),
        ),
    )
    cap.render_order = 2

    # Put the plane geometry, which lies in z = 0, onto the clipping plane
    cap.local.rotation = la.quat_from_vecs((0, 0, 1), plane[:3])
    cap.local.position = [plane[3] * c for c in plane[:3]]
    scene.add(cap)

camera = gfx.PerspectiveCamera(50, 16 / 9)
camera.show_object(knot, view_dir=(-1, -1, -1), scale=0.4)
scene.add(gfx.AmbientLight(), camera.add(gfx.DirectionalLight()))

controller = gfx.OrbitController(camera, register_events=renderer)


def animate():
    rot = la.quat_from_euler((0.006, 0.01), order="XY")
    knot.local.rotation = la.quat_mul(rot, knot.local.rotation)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    loop.run()
