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
camera.show_rect(-0.5, n + 0.5, 0, 2)
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#000"))


plane = gfx.plane_geometry(1, dy)
for i in range(n):
    alpha = (i + 1) / n
    # Blend
    m = gfx.Mesh(
        plane, gfx.MeshBasicMaterial(color="#fff", opacity=alpha, blending="normal")
    )
    m.local.x = i
    m.local.y = 0
    scene.add(m)
    # Dither
    m = gfx.Mesh(
        plane, gfx.MeshBasicMaterial(color="#fff", opacity=alpha, blending="dither")
    )
    m.local.x = i
    m.local.y = 1 * dy
    scene.add(m)

canvas.request_draw(lambda: renderer.render(scene, camera))


if __name__ == "__main__":
    print(__doc__)
    loop.run()
