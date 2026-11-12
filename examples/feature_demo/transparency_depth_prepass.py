"""
Transparency with a depth pre-pass
==================================

A transparent object also shows its own hidden surfaces: the inner and back
faces of the left torus knot shine through its front. A common fix is a depth
pre-pass: first render the object with ``color_write=False``, so that it only
writes its depth, and then render it transparently with ``depth_compare="<="``,
so that only its front-most surface is blended in (right).
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx
import pylinalg as la


canvas = RenderCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#e0e0e0", "#606060"))

geometry = gfx.torus_knot_geometry(1, 0.3, 128, 32)


def transparent_material(**kwargs):
    return gfx.MeshPhongMaterial(
        color="#4cc9f0", opacity=0.5, alpha_mode="blend", **kwargs
    )


# Left: just transparent
knot1 = gfx.Mesh(geometry, transparent_material())
knot1.local.x = -2.2

# Right: a depth-only pre-pass, then the transparent pass on top of it
knot2 = gfx.Mesh(geometry, transparent_material(depth_compare="<="))
prepass = gfx.Mesh(geometry, transparent_material(color_write=False, depth_write=True))
prepass.render_order = -1  # render before the transparent pass
knot2.add(prepass)
knot2.local.x = 2.2

scene.add(knot1, knot2)

camera = gfx.PerspectiveCamera(50, 16 / 9)
camera.show_object(scene, view_dir=(0, 0, -1))
scene.add(camera.add(gfx.DirectionalLight()))
scene.add(gfx.AmbientLight())

controller = gfx.OrbitController(camera, register_events=renderer)


def animate():
    rot = la.quat_from_euler((0.005, 0.01), order="XY")
    for knot in (knot1, knot2):
        knot.local.rotation = la.quat_mul(rot, knot.local.rotation)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    print(__doc__)
    canvas.request_draw(animate)
    loop.run()
