"""
Reference Color
===============

This example draws squares of reference colors. These can be compared to
similar output from e.g. Matplotlib.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx

canvas = RenderCanvas()
renderer = gfx.WgpuRenderer(canvas, gamma_correction=1.0)

n = 32
dy = n / 4

camera = gfx.OrthographicCamera()
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#000"))


plane = gfx.plane_geometry(1, dy)
for i in range(n):
    alpha = (i + 1) / n
    # Blend
    m = gfx.Mesh(
        plane, gfx.MeshBasicMaterial(color="#fff", opacity=alpha, alpha_mode="blend")
    )
    m.local.x = i
    m.local.y = 0
    scene.add(m)
    # Dither bayer
    m = gfx.Mesh(
        plane,
        gfx.MeshBasicMaterial(color="#fff", opacity=alpha, alpha_mode="bayer"),
    )
    m.local.x = i
    m.local.y = 1 * dy
    scene.add(m)
    # Dither
    m = gfx.Mesh(
        plane, gfx.MeshBasicMaterial(color="#fff", opacity=alpha, alpha_mode="dither")
    )
    m.local.x = i
    m.local.y = 2 * dy
    scene.add(m)

camera.show_object(scene, match_aspect=True)
canvas.request_draw(lambda: renderer.render(scene, camera))


if __name__ == "__main__":
    print(__doc__)
    loop.run()
