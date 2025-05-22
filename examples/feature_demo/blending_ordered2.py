"""
Blending ordered2
=================

The 'ordered2' was a previously supported blend mode (when blendingw as defined on the renderer).
This example mimics it's behaviour.

This example draws a red (top) and a green (bottom) circle. The circles
are opaque but have a transparent edge. I.e. the objects have both
opaque and transparent fragments, which is a case that can result in
wrong blending.

The green circle is drawn after the red one, but it is also behind the red one.
This means that without proper care, the edge of the red circle hides
the green circle.

By first rendering only the opaque fragments, and then rendering the transparent
fragments, this can be fixed. To do this, we use the alpha test.
"""

from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx


canvas = RenderCanvas()
renderer = gfx.WgpuRenderer(canvas)
scene = gfx.Scene()

scene.add(gfx.Background.from_color("#000"))

points1 = gfx.Points(
    gfx.Geometry(positions=[[0, 0, 0]]),
    gfx.PointsMarkerMaterial(
        size_space="model",
        edge_mode="outer",
        size=1.5,
        edge_width=0.2,
        color="#f00",
        edge_color="#fff7",
    ),
)
scene.add(points1)

points2 = gfx.Points(
    gfx.Geometry(positions=[[0, -1, -1]]),
    gfx.PointsMarkerMaterial(
        size_space="model",
        edge_mode="outer",
        size=1.5,
        edge_width=0.2,
        color="#0f0",
        edge_color="#fff7",
    ),
)
scene.add(points2)

camera = gfx.PerspectiveCamera(0)
camera.show_object(scene)


def animate():
    # Draw opaque fragments (alpha > 0.999)
    for ob in scene.iter():
        if ob.material:
            ob.material.alpha_test = 0.999
            ob.material.depth_write = True
    renderer.render(scene, camera, flush=False)

    # Draw transparent fragments (alpha < 0.999)
    for ob in scene.iter():
        if ob.material:
            ob.material.alpha_test = -0.999
            ob.material.depth_write = False
    renderer.render(scene, camera, flush=False)

    renderer.flush()


canvas.request_draw(animate)

if __name__ == "__main__":
    loop.run()
